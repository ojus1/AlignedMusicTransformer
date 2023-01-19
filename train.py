from model import MusicTransformer
from model_rwkv import MusicTransformerRWKV
import custom
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from data import Data
from midi_processor_fixed.transposition_aug import augment_batch

import utils
import datetime
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter


# set config
parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


# load data
dataset = Data(config.pickle_dir)
print(dataset)


# load model
learning_rate = config.l_r

# define model
mt = MusicTransformerRWKV(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            num_layer=config.num_layers,
            max_seq=config.max_seq,
            dropout=config.dropout,
            debug=config.debug, loader_path=config.load_path
)
mt.to(config.device)
opt = optim.Adam(mt.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)

# multi-GPU set
if torch.cuda.device_count() > 1:
    single_mt = mt
    mt = torch.nn.DataParallel(mt, output_device=torch.cuda.device_count()-1)
else:
    single_mt = mt

# init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
    'bucket':  LogitsBucketting(config.vocab_size)
})

print(mt)
print('| Summary - Device Info : {}'.format(torch.cuda.device))

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+config.experiment+'/'+current_time+'/train'
eval_log_dir = 'logs/'+config.experiment+'/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)

best_acc = 0.0

# Train Start
print(">> Train start...")
idx = 0
for e in range(config.epochs):
    print(">>> [Epoch was updated]")
    for b in range(len(dataset.files) // config.batch_size):
        scheduler.optimizer.zero_grad()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq)
            # pitch transposition and time stretch augmentation
            batch_x, batch_y = augment_batch(batch_x, batch_y, config.token_eos, config.token_sos, shift_range=3)
            batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
            batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
        except IndexError:
            continue

        start_time = time.time()
        mt.train()
        sample = mt.forward(batch_x)
        metrics = metric_set(sample, batch_y)
        loss = metrics['loss']
        loss.backward()
        scheduler.step()
        end_time = time.time()

        if config.debug:
            print("[Loss]: {}".format(loss))

        train_summary_writer.add_scalar('loss', metrics['loss'], global_step=idx)
        train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=idx)
        train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=idx)
        train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=idx)

        # result_metrics = metric_set(sample, batch_y)
        if b % 512 == 0:
            single_mt.eval()
            preds = []
            gt = []
            with torch.inference_mode():
                for _ in range(512 // config.batch_size):
                    eval_x, eval_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq, 'eval')
                    eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
                    eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)

                    eval_preiction = single_mt.forward(eval_x)
                    preds.append(eval_preiction.cpu())
                    gt.append(eval_y.cpu())

                    del eval_preiction
                    del eval_x
                    del eval_y
                    
                    torch.cuda.empty_cache()

                preds = torch.cat(preds, dim=0)
                gt = torch.cat(gt, dim=0)
                # print(preds.shape, gt.shape)
            eval_metrics = metric_set(preds, gt)
            # if b == 0:
            #     train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
            #     train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
            #     for i, weight in enumerate(weights):
            #         attn_log_name = "attn/layer-{}".format(i)
            #         utils.attention_image_summary(
            #             attn_log_name, weight, step=idx, writer=eval_summary_writer)

            if best_acc < eval_metrics["accuracy"]:
                best_acc = eval_metrics["accuracy"]
                torch.save(single_mt.state_dict(), args.model_dir+'/acc-{}.pth'.format(best_acc))

            eval_summary_writer.add_scalar('loss', eval_metrics['loss'], global_step=idx)
            eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=idx)
            eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=idx)

            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))
        torch.cuda.empty_cache()
        idx += 1

        # switch output device to: gpu-1 ~ gpu-n
        sw_start = time.time()
        if torch.cuda.device_count() > 1:
            mt.output_device = idx % (torch.cuda.device_count() -1) + 1
        sw_end = time.time()
        if config.debug:
            print('output switch time: {}'.format(sw_end - sw_start) )

torch.save(single_mt.state_dict(), args.model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()


