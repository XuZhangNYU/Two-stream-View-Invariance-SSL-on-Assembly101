import argparse
import random
from torch import optim
import pandas as pd
from utils import *
from dataset_baseline_tripleview import *
from model import *
import wandb
import random
from torch.nn import Transformer

from collections import OrderedDict
from torch import nn
wandb.login()

# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_c2f_ensemble_output(outp, weights):
    ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

    for i, outp_ele in enumerate(outp[1]):
        upped_logit = F.upsample(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
        ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)

    return ensemble_prob



################## Trainer (change loss)
class Trainer:
    def __init__(self):
        set_seed(seed)
        self.model = C2F_TCN(config.feature_size, config.num_class)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2048, nhead=32), num_layers=6) #d_model=1024
        # self.reseter = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2048, nhead=32), num_layers=10) 
 
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        assert self.num_classes > 0, "wrong class numbers"
        print('Model Size: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.es = EarlyStop(patience=args.patience)
        # ~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!
        transformer_weight_name = '/ssl_TRIPLE_VIC_epoch-460-v3.model'
        # c2f_weight_name = '/c2f_epoch-best-batch64-large_step-chunk20-maxfram1200-v3.model'
        prefixed_stat_dict = torch.load(model_dir + transformer_weight_name, map_location=device)

        new_state_dict = OrderedDict()
        for k, v in prefixed_stat_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v

        # Load the modified state_dict
        self.transformer.load_state_dict(new_state_dict)

        for name, param in self.transformer.named_parameters():
            param.requires_grad = False

        # self.model.load_state_dict(torch.load(model_dir + c2f_weight_name, map_location=device)['model_state_dict'])


    def train(self, save_dir, num_epochs):
        if (args.use_check_point != "None"):
            try:
                pretrained_weights_object = torch.load(args.use_check_point, map_location=device)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!
                # self.model.load_state_dict(pretrained_weights_object['model_state_dict'])
                print('learning rate is ',  config.learning_rate)
                # print('loaded model weights from checkpoint successfully')
                # optimizer.load_state_dict(pretrained_weights_object['optimizer_state_dict'])
                # print('use pretrained optimizer state successfully')
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = config.learning_rate
                #     print('optimizer spec are', param_group['lr'])
                print("NOTIFICATION: !!!!!!!")
                print("loaded check point at, ", args.use_check_point)
            except:
                print("no such pretrained weight exists!!!!! save a new checkpoint instead !!!")
                # torch.save({
                #     'total_epochs': num_epochs,
                #     'current_epoch' : 0, 
                #     'model_state_dict': self.model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),  # Optional, if you want to resume training
                # }, save_dir + "/epoch-latest" + ".model")

        self.model.train()
        self.model.to(device)
        self.transformer.train()
        self.transformer.to(device)
        # self.reseter.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        best_score = -10000
        for epoch in range(num_epochs):
            correct, total, nums = 0, 0, 0
            epoch_loss, ce_loss, smooth_loss = 0.0, 0.0, 0.0
            for i, item in enumerate(train_loader):
                nums += 1
                samples = item[0].to(device)
                count = item[1].to(device)
                labels = item[2].to(device)
                src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                src_mask = src_mask.to(device)

                src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

                transformer_out = self.transformer(samples)
                # Not residual Connection
                residual_out = transformer_out + samples

                # if(epoch % 2 == 0):
                #     foo = self.reseter(samples[:8,:,:])


                outputs_list = self.model(residual_out.permute(0, 2, 1))
                
                outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)

                outp_wo_softmax = torch.log(
                    outputs_ensemble + 1e-10)  # log is necessary because ensemble gives softmax output
                ce_l = self.ce(outp_wo_softmax, labels)
                mse_l = 0.17 * torch.mean(
                    torch.clamp(self.mse(outp_wo_softmax[:, :, 1:], outp_wo_softmax.detach()[:, :, :-1]),
                                min=0, max=16) * src_msk_send[:, :, 1:])
                loss = ce_l + mse_l
                ce_loss += ce_l.item()
                smooth_loss += mse_l.item()
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = torch.argmax(outputs_ensemble, dim=1)
                correct += float(torch.sum((pred == labels) * src_mask).item())
                total += float(torch.sum(src_mask).item())

            scheduler.step()
            pr_str = "[epoch %d]: loss = %.3f, ce = %.3f, sm = %.3f, acc = %.3f" % \
                     (epoch + 1, epoch_loss / nums, ce_loss / nums, smooth_loss / nums, float(correct) / total)
            print(pr_str)
 
            wandb.log({"accuracy": float(correct) / total, "loss": epoch_loss / nums})

            if (args.use_check_point != "None"):
                # try assume we have pretrained weight loaded, else we use 0 for pretrained_weights_object['current_epoch'] 
                try:
                    current_epoch = pretrained_weights_object['current_epoch'] + epoch
                except:
                    current_epoch = epoch
            else:
                current_epoch = epoch

            if (epoch + 1) % 25 == 0:
                torch.save({
                    'total_epochs': num_epochs,
                    'current_epoch' : current_epoch, 
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),  # Optional, if you want to resume training
                }, save_dir + "/final_head_TRIPLE_recipe_victory_walked460_epoch-" + str(epoch + 1) + ".model")

                
                for param_group in optimizer.param_groups:
                    print('learning rate is ',  param_group['lr']) 

            test_score = self.test(epoch)
            if test_score > best_score:
                best_score = test_score
                torch.save({
                    'total_epochs': num_epochs,
                    'current_epoch' : current_epoch, 
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),  # Optional, if you want to resume training
                }, save_dir + "/final_head_TRIPLE_recipe_victory_walked460_epoch-best" + ".model")
                print("Save for the best model")
            # if self.es.step(loss=None, acc=test_score, criterion=lambda x1, x2: x2):
            #     print("Early stop!")
            #     exit(0)

    def test(self, epoch):
        self.model.eval()
        self.transformer.eval()

        epoch_loss, ce_loss, smooth_loss = 0.0, 0.0, 0.0
        nums = 0
        with torch.no_grad():
            for i, item in enumerate(test_loader):
                nums += 1
                samples = item[0].to(device)
                count = item[1].to(device)
                labels = item[2].to(device)
                src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                src_mask = src_mask.to(device)
                src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

                transformer_out = self.transformer(samples)
                residual_out = transformer_out + (samples)
                outputs_list = self.model(residual_out.permute(0, 2, 1))


                outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)

                outp_wo_softmax = torch.log(outputs_ensemble + 1e-10)
                ce_l = self.ce(outp_wo_softmax, labels)
                mse_l = 0.17 * torch.mean(
                    torch.clamp(self.mse(outp_wo_softmax[:, :, 1:], outp_wo_softmax.detach()[:, :, :-1]),
                                min=0, max=16) * src_msk_send[:, :, 1:])
                loss = ce_l + mse_l
                ce_loss += ce_l.item()
                smooth_loss += mse_l.item()
                epoch_loss += loss.item()
                postprocessor(outputs_ensemble, item[5], labels, count)

            pr_str = "***[epoch %d]***: loss = %.3f, ce = %.3f, sm = %.3f" % (epoch + 1, epoch_loss / nums, ce_loss / nums, smooth_loss / nums)
            print(pr_str)

            path = os.path.join(results_dir, "tempt" + os.path.basename(__file__))
            if not os.path.exists(path):
                os.mkdir(path)
            if not os.path.exists(path + '_gt'):
                os.mkdir(path + '_gt')
            postprocessor.dump_to_directory(path, '_gt')
            final_edit_score, acc, overlap_scores = calculate_mof(path + '_gt', path, config.back_gd)
            postprocessor.start()

            results = {}
            # action: standard metrics
            results['f1_10'] = overlap_scores[0]
            results['f1_25'] = overlap_scores[1]
            results['f1_50'] = overlap_scores[2]
            results['f_acc'] = acc
            results['edit'] = final_edit_score
            results['total_score'] = results['f1_10'] + results['f1_25'] + results['f1_50'] + results['f_acc'] + \
                                     results['edit']

            print("---[epoch %d]---: tst edit = %.2f, f1_10 = %.2f, f1_25 = %.2f, f1_50 = %.2f, acc = %.2f, total = %.2f "
                % (epoch + 1, results['edit'], results['f1_10'], results['f1_25'], results['f1_50'],
                   results['f_acc'], results['total_score']))
            
            wandb.log({"edit score": float(results['edit']), "f1 score 10%": results['f1_10'], "f1 score 25%": results['f1_25'], "f1 score 50%": results['f1_50']})


        self.model.train()
        self.transformer.train()

        return results['total_score']

    def predict(self, model_dir, results_dir, eval_loader, postprocessor_eval):

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-best" + ".model", map_location=device))

            for i, item in enumerate(eval_loader):
                samples = item[0].to(device)
                count = item[1].to(device)
                labels = item[2].to(device)

                transformer_out = self.transformer(samples)
                residual_out = transformer_out + (samples)
                outputs_list = self.model(residual_out.permute(0, 2, 1))


                outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)
                postprocessor_eval(outputs_ensemble, item[5], labels, count, item[7].to(device), item[8],
                                   item[3].to(device))

            postprocessor_eval.dump_to_directory(results_dir)
            final_edit_score, acc, overlap_scores = calculate_mof(results_dir + '_gt', results_dir, config.back_gd)
            postprocessor_eval.start()

            results = {}
            # action: standard metrics
            results['f1_10'] = overlap_scores[0]
            results['f1_25'] = overlap_scores[1]
            results['f1_50'] = overlap_scores[2]
            results['f_acc'] = acc
            results['edit'] = final_edit_score
            results['total_score'] = results['f1_10'] + results['f1_25'] + results['f1_50'] + results['f_acc'] + \
                                     results['edit']

            print("---[Eval]---: tst edit = %.2f, f1_10 = %.2f, f1_25 = %.2f, f1_50 = %.2f, acc = %.2f, total = %.2f "
                % (results['edit'], results['f1_10'], results['f1_25'], results['f1_50'],
                   results['f_acc'], results['total_score']))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--feature_path', type=str, default='/mnt/data/zhanzhong/assembly/')
parser.add_argument('--dataset', default="assembly")
parser.add_argument('--split', default='train')  # or 'train_val'
parser.add_argument('--seed', default='42')
parser.add_argument('--test_aug', type=int, default=0)
parser.add_argument('--patience', type=int, default=50)
# !!!!!!!!!!!!!!!!
parser.add_argument("--use_check_point", type=str, default="None")

args = parser.parse_args()

seed = int(args.seed)

set_seed(seed)
# VIEWS = ['C10095_rgb', 'C10115_rgb', 'C10118_rgb', 'C10119_rgb', 'C10379_rgb', 'C10390_rgb', 'C10395_rgb', 'C10404_rgb',
            #  'HMC_21176875_mono10bit', 'HMC_84346135_mono10bit', 'HMC_21176623_mono10bit', 'HMC_84347414_mono10bit',
            #  'HMC_21110305_mono10bit', 'HMC_84355350_mono10bit', 'HMC_21179183_mono10bit', 'HMC_84358933_mono10bit']

# VIEWS = ['HMC_21179183_mono10bit', 'HMC_84358933_mono10bit']

# !!!!!!!!!!!!!!!1
VIEWS = ['C10118_rgb', 'C10119_rgb', 'HMC_21179183_mono10bit', 'HMC_84358933_mono10bit']


config = dotdict(
    epochs=500,
    dataset=args.dataset,
    feature_size=2048,
    gamma=0.5,
    step_size=40,
    split=args.split)

if args.dataset == "assembly":
    # config.chunk_size = 20
    # config.max_frames_per_video = 1200
    config.chunk_size = 20
    config.max_frames_per_video = 1200
    config.learning_rate = 1e-3
    config.weight_decay = 1e-4
    config.batch_size = 64
    config.num_class = 202
    config.back_gd = ['Background']
    config.ensem_weights = [1, 1, 1, 1, 1, 1]
    if args.action == 'predict':
        if int(args.test_aug):
            config.chunk_size = list(range(10, 31, 7))
            config.weights = np.ones(len(config.chunk_size))
            config.weights = np.ones(len(config.chunk_size))
        else:
            # config.chunk_size = [20]
            config.chunk_size = [20]
            config.weights = [1]
else:
    print('not defined yet')
    exit(1)

run = wandb.init(
    project="my-awesome-project-1st",    # Specify your project
    config={                         # Track hyperparameters and metadata
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
    },
)

TYPE = '/c2f_{}'.format(args.seed)
model_dir = "/scratch/xz4863/synchranization_all_u_need_results/" + args.dataset + "/" + args.split + TYPE #output model path
results_dir = "/scratch/xz4863/synchranization_all_u_need_results/" + args.dataset + "/" + args.split + TYPE #output results path

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# ！！！！！！！！！！！！！！！！ change paths
vid_list_path = "/home/xz4863/scratch/synchronization/data/coarse-annotations/coarse_splits/" 
gt_path = "/home/xz4863/scratch/synchronization/data/coarse-annotations/coarse_labels/" 
mapping_file = "/home/xz4863/scratch/synchronization/data/coarse-annotations/actions.csv" 
features_path = args.feature_path

config.features_path = features_path
config.gt_path = gt_path
config.VIEWS = VIEWS

actions = pd.read_csv(mapping_file, header=0,
                      names=['action_id', 'verb_id', 'noun_id', 'action_cls', 'verb_cls', 'noun_cls'])
actions_dict, label_dict = dict(), dict()
for _, act in actions.iterrows():
    actions_dict[act['action_cls']] = int(act['action_id'])
    label_dict[int(act['action_id'])] = act['action_cls']

num_classes = len(actions_dict)
assert num_classes == config.num_class

############################dataloader
def _init_fn(worker_id):
    np.random.seed(int(seed))

###########################postprocessor
postprocessor = PostProcess(config, label_dict, actions_dict, gt_path).to(device)

trainer = Trainer()
if args.action == "train":
    train_dataset = AugmentDataset(config, fold=args.split, fold_file_name=vid_list_path, actions_dict=actions_dict, zoom_crop=(0.5, 2))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=1, collate_fn=collate_fn_override,
                                               worker_init_fn=_init_fn)

    test_dataset = AugmentDataset(config, fold='val', fold_file_name=vid_list_path, actions_dict=actions_dict, zoom_crop=(0.5, 2))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False,
                                              pin_memory=True, num_workers=1, collate_fn=collate_fn_override,
                                              worker_init_fn=_init_fn)

    trainer.train(model_dir, num_epochs=config.epochs)

if args.action == "predict":
    eval_dataset = AugmentDataset_test(config, fold='val', fold_file_name=vid_list_path, actions_dict=actions_dict, chunk_size=config.chunk_size)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=config.batch_size, shuffle=False,
                                              pin_memory=True, num_workers=1, collate_fn=collate_fn_override_test,
                                              worker_init_fn=_init_fn)
    postprocessor_eval = PostProcess_test(config.weights, label_dict, actions_dict, gt_path).to(device)

    if not os.path.exists(os.path.join(results_dir, 'prediction{}'.format('_aug' if int(args.test_aug) else ''))):
        os.makedirs(os.path.join(results_dir, 'prediction{}'.format('_aug' if int(args.test_aug) else '')))
    trainer.predict(model_dir, os.path.join(results_dir, 'prediction{}'.format('_aug' if int(args.test_aug) else '')),
                    eval_loader, postprocessor_eval)

