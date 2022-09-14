#coding=utf8
import sys, os, time, json, gc, datetime, csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.args import init_args
from utils.hyperparams import hyperparam_path
from utils.initialization import *
from utils.example import Example
from utils.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import Text2SQL
from tensorboardX import SummaryWriter
from utils.constants import RELATIONS, RELATIONS_INDEX
from utils.logger import Logger
from utils.evaluator import acc_token
from tqdm import tqdm

    
def decode(dataset, output_path, acc_type='sql', use_checker=False, use_standard=False):
    assert acc_type in ['beam', 'ast', 'sql']
    model.eval()
    all_hyps = []
    table_predicts, column_predicts, table_labels, column_labels, table_sqls, column_sqls = [], [], [], [], [], []
    _correct, _total, _total1 = 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
            hyps, labels, sql_schemas = model.parse(current_batch, args.beam_size, dataset[i: i + args.batch_size], use_standard=use_standard)
            all_hyps.extend(hyps)
        acc = evaluator.acc(all_hyps, dataset, output_path, acc_type=acc_type, etype='match', use_checker=use_checker)
    return acc


if __name__ == '__main__':
    
    args = init_args(sys.argv[1:])
    set_random_seed(args.seed)
    logger = Logger(args.logdir, vars(args))
    device = set_torch_device(args.device)
    logger.log("Initialization finished ...")
    logger.log("Random seed is set to %d" % (args.seed))
    
    # load dataset and vocabulary
    start_time = time.time()

    Example.configuration(plm=args.plm, method=args.model)
    dev_dk_dataset = Example.load_dataset(args.dev_dk_path) if args.dev_dk_path != "" else None
    dev_syn_dataset = Example.load_dataset(args.dev_syn_path) if  args.dev_syn_path != "" else None
    train_dataset, dev_dataset = Example.load_dataset(args.train_path), Example.load_dataset(args.dev_path)

    logger.log("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
    logger.log("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
    sql_trans, evaluator = Example.trans, Example.evaluator
    args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relation_vocab)

    # model init, set optimizer
    model = Text2SQL(args, sql_trans).to(device)

    if args.read_model_path and args.read_model_path != 'none':
        check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
        model.load_state_dict(check_point['model'])
        logger.log("Load saved model from path: %s" % (args.read_model_path))
    else:
        json.dump(vars(args), open(os.path.join(args.logdir, 'params.json'), 'w'), indent=4)
        if args.plm is None:
            ratio = Example.word2vec.load_embeddings(model.encoder.input_layer.word_embed, Example.word_vocab, device=device)
            logger.log("Init model and word embedding layer with a coverage %.2f" % (ratio))

    if args.training:
        num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
        num_warmup_steps = int(num_training_steps * args.warmup_ratio)
        logger.log('Total training steps: %d;\t Warmup steps: %d' % (num_training_steps, num_warmup_steps))
        
        optimizer, scheduler = set_optimizer(model, args, num_warmup_steps, num_training_steps)
        start_epoch, nsamples, best_result = 0, len(train_dataset), {'dev_acc': 0.}
        train_index, step_size = np.arange(nsamples), args.batch_size // args.grad_accumulate
        
        logger.log('Start training ......')

        for i in range(start_epoch, args.max_epoch):
            start_time = time.time()
            epoch_loss, epoch_gp_loss, count = 0, 0, 0
            np.random.shuffle(train_index)
            model.train()
            table_predicts, column_predicts, table_labels, column_labels = [], [], [], []
            for j in range(0, nsamples, step_size):
                count += 1
                cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
                current_batch = Batch.from_example_list(cur_dataset, device, train=True, smoothing=args.smoothing)
                loss = model(current_batch, cur_dataset, train=True, use_standard=args.use_standard)  # see utils/batch.py for batch elements

                epoch_loss += loss.item()
                loss.backward()
                if j % 1000 == 0:
                    logger.write_metrics({'step': j, 'loss': loss.item()},'step')
                
                if count == args.grad_accumulate or j + step_size >= nsamples:
                    count = 0
                    model.pad_embedding_grad_zero()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if i < args.eval_after_epoch:
                continue

            start_time = time.time()
            dev_acc = decode(dev_dataset, os.path.join(args.logdir, 'dev.iter' + str(i)), acc_type='sql')
            logger.log('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_acc))
            
            if args.dev_syn_path != "":
                start_time = time.time()
                dev_syn_acc = decode(dev_syn_dataset, os.path.join(args.logdir, 'dev_syn.iter' + str(i)), acc_type='sql')
                logger.log('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_syn_acc))

            if args.dev_dk_path != "":
                start_time = time.time()
                dev_dk_acc = decode(dev_dk_dataset, os.path.join(args.logdir, 'dev_dk.iter' + str(i)), acc_type='sql')
                logger.log('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_dk_acc))

            if dev_acc > best_result['dev_acc']:
                best_result['dev_acc'], best_result['iter'] = dev_acc, i
                torch.save({
                    'epoch': i, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, open(os.path.join(args.logdir, 'model.bin'), 'wb'))
                logger.log('NEW BEST MODEL: \tEpoch: %d\tDev acc: %.4f' % (i, dev_acc))

        logger.log('FINAL BEST RESULT: \tEpoch: %d\tDev acc: %.4f' % (best_result['iter'], best_result['dev_acc']))

    if args.testing:
        start_time = time.time()
        dev_acc = decode(dev_dataset, output_path=os.path.join(args.logdir, 'dev.eval'), acc_type='sql')
        print(dev_acc)
        dev_acc_checker = decode(dev_dataset, output_path=os.path.join(args.logdir, 'dev.eval.checker'), acc_type='sql', use_checker=True, use_standard=False)
        print(dev_acc_checker)
        logger.log("Evaluation costs %.2fs ; Dev dataset exact match/checker/beam acc is %.4f/%.4f ." % (time.time() - start_time, dev_acc, dev_acc_checker))
