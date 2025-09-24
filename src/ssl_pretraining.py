import argparse
import random
from torch import optim
import pandas as pd
from utils import *
from dataset_ssl import *
from model import *
import wandb
from torch.nn import Transformer
import sys

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

if args.ssl_method == 'our_method':
    class SynthViewsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformerEncode_gate = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=4096, nhead=8),  # d_model=2048
                num_layers=3
            )

            self.input_proj_up = nn.Linear(4096, 6144)  # Project input from 4096 → 2048

            self.transformerEncode_3d = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=6144, nhead=16),  # d_model=2048
                num_layers=6
            )

            self.input_proj_down = nn.Linear(6144, 2048)  # Project input from 2048 → 4096

            self.transformerEncode_out = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=2048, nhead=32),  # d_model=2048
                num_layers=1
            )


        def forward(self, x):
            """
            x: shape (seq_len, batch_size, 4096)
            """
            x = self.transformerEncode_gate(x) # (seq_len, batch_size, 2048)
            x = self.input_proj_up(x)  # (seq_len, batch_size, 2048)
            x = self.transformerEncode_3d(x) # (seq_len, batch_size, 2048)
            x = self.input_proj_down(x)  # (seq_len, batch_size, 2048)
            x = self.transformerEncode_out(x) # (seq_len, batch_size, 2048)

            return x
elif args.ssl_method == "ablation_simple_3d":
    class SynthViewsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformerEncode_gate = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=4096, nhead=32),  # d_model=2048
                num_layers=10
            )
            self.input_proj_down = nn.Linear(4096, 2048)  # Project input from 2048 → 4096


        def forward(self, x):
            """
            x: shape (seq_len, batch_size, 4096)
            """
            x = self.transformerEncode_gate(x) # (seq_len, batch_size, 2048)
            x = self.input_proj_down(x)  # (seq_len, batch_size, 2048)

            return x

################## Trainer (change loss)
class Trainer:
    def __init__(self):
        set_seed(seed)
        self.C2F_model = C2F_TCN(config.feature_size, config.num_class) 

        if (args.ssl_method == 'our_method') or (args.ssl_method == 'ablation_simple_3d'):
            self.synth_views_model = SynthViewsModel() #d_model=1024
            self.invar_view_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2048, nhead=32), num_layers=6)#d_model=1024
            # self.invar_view_model_dual = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2048, nhead=32), num_layers=6)#d_model=1024
            # for name, param in self.invar_view_model_dual.named_parameters():
            #     param.requires_grad = False

            self.criterion_mse = nn.MSELoss()
            self.criterion_l1 = nn.L1Loss()
            self.criterion_var_loss = self.variance_loss
            self.optimizer_invar = optim.SGD(self.invar_view_model.parameters(), lr=0.0001)
            self.optimizer_synth = optim.SGD(self.synth_views_model.parameters(), lr=0.0001)
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs!")
                self.synth_views_model = torch.nn.DataParallel(self.synth_views_model)
                self.invar_view_model = torch.nn.DataParallel(self.invar_view_model)
        
            self.synth_views_model = self.synth_views_model.cuda()
            self.invar_view_model = self.invar_view_model.cuda()
            print('Model Size: {}'.format(sum(p.numel() for p in self.invar_view_model.parameters())))
        elif args.ssl_method == 'sync_kd':             
            self.model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2048, nhead=32), num_layers=6) #d_model=1024
            self.criterion = nn.MSELoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
            print('Model Size: {}'.format(sum(p.numel() for p in self.model.parameters())))

        self.num_classes = num_classes
        assert self.num_classes > 0, "wrong class numbers"
        self.es = EarlyStop(patience=args.patience)


    @torch.no_grad()
    def update_moving_average(self, online_encoder, target_encoder, m):
        for param_online, param_target in zip(online_encoder.parameters(), target_encoder.parameters()):
            param_target.data = param_target.data * m + param_online.data * (1.0 - m)
            param_target.requires_grad = False


    def variance_loss(self, z, eps=1e-6):
        # z: (batch_size, feature_dim)
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    if (args.ssl_method == 'our_method') or (args.ssl_method == 'ablation_simple_3d'):
        def train(self, save_dir, num_epochs):
            self.synth_views_model.train()
            self.synth_views_model.to(device)
            self.invar_view_model.train()
            self.invar_view_model.to(device)
            # self.invar_view_model_dual.train()
            # self.invar_view_model_dual.to(device)

            scheduler_synth = torch.optim.lr_scheduler.StepLR(self.optimizer_synth, step_size=config.step_size, gamma=config.gamma)
            scheduler_invar = torch.optim.lr_scheduler.StepLR(self.optimizer_invar, step_size=config.step_size, gamma=config.gamma)
        
            best_score = 10000
            for epoch in range(num_epochs):
                running_loss = 0.0
                running_gt_loss = 0.0
                correct, total, nums = 0, 0, 0
                loss_recon_epoch = 0.0
                loss_invar_epoch = 0.0 
                epoch_loss, mse_l, kld_l, y1_l, y2_l, y3_l, y4_l, y5_l, ce_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for i, item in enumerate(train_loader):
                    self.optimizer_invar.zero_grad()
                    self.optimizer_synth.zero_grad()

                    nums += 1
                    samples_src = item[0].to(device)
                    samples_tgt = item[1].to(device)
                    count = item[2].to(device)
                    labels = item[3].to(device)
                    src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                    src_mask = src_mask.to(device)
                    src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

                    cat_sync_vid = torch.cat([samples_src + 1, samples_tgt - 1], dim=2)               # shape: (4096,)
                    synth_rep = self.synth_views_model(cat_sync_vid) 

                    view_invar_rep_ego = self.invar_view_model(samples_src) # shape: (4096,)
                    view_invar_rep_ego += samples_src

                    view_invar_rep_exo = self.invar_view_model(samples_tgt) # shape: (4096,)
                    view_invar_rep_exo += samples_tgt

                    gt_loss = self.criterion_mse(samples_src, samples_tgt)
                    loss_recon = self.criterion_mse(view_invar_rep_ego, synth_rep) + self.criterion_mse(view_invar_rep_exo, synth_rep)
                    loss_invar = ((100 + epoch)/ (100 + (epoch ** 1.18))) * self.criterion_mse(view_invar_rep_ego, view_invar_rep_exo)
                    loss = loss_recon + loss_invar

                    loss_recon_epoch += loss_recon.item()
                    loss_invar_epoch += loss_invar.item() 

                    loss.backward()
                    self.optimizer_synth.step()
                    self.optimizer_invar.step()

                        # self.update_moving_average(self.invar_view_model, self.invar_view_model_dual, m=0.9)
                scheduler_invar.step()
                scheduler_synth.step()

                pr_str = "[epoch %d]: loss_mse = %.3f, loss_invar = %.3f" % (epoch+1, loss_recon/nums, loss_invar_epoch/nums)
                print(pr_str)

                wandb.log({"loss_recon": loss_recon_epoch/nums, "loss_invar": loss_invar_epoch/nums})

                if (epoch + 1) % 10 == 0:
                    torch.save(self.invar_view_model.state_dict(), save_dir + "/ssl_vic_epoch-" + str(epoch + 1) + ".model")

                test_score = self.test(epoch)
                if test_score > best_score:
                    best_score = test_score
                    torch.save(self.invar_view_model.state_dict(), save_dir + "/ssl_vic_epoch-best" + ".model")
                    print("Save for the best model")
    elif args.ssl_method == 'sync_kd':
        def train(self, save_dir, num_epochs):
            self.model.train()
            self.model.to(device)
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)
            best_score = 10000
            for epoch in range(num_epochs):
                running_loss = 0.0
                running_gt_loss = 0.0
                correct, total, nums = 0, 0, 0
                epoch_loss, mse_l, kld_l, y1_l, y2_l, y3_l, y4_l, y5_l, ce_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for i, item in enumerate(train_loader):
                    #print("Iter [%d/%d]" % (nums,len(train_loader)))
                    self.optimizer.zero_grad()
                    nums += 1
                    samples_src = item[0].to(device)
                    samples_tgt = item[1].to(device)
                    count = item[2].to(device)
                    labels = item[3].to(device)
                    src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                    src_mask = src_mask.to(device)

                    src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

                    output = self.model(samples_tgt)
                    output = output + samples_tgt #SKIP CONNECTIONS
                    gt_loss = self.criterion(samples_tgt, samples_src)
                    loss = self.criterion(output, samples_src)
                    loss.backward()
                    self.optimizer.step()
                    running_gt_loss += gt_loss.item()
                    running_loss += loss.item()


                scheduler.step()
                
                pr_str = "[epoch %d]: loss = %.3f, running_loss = %.3f" % (epoch+1, loss/nums, running_loss/nums)
                print(pr_str)

                wandb.log({"running_gt_loss": running_gt_loss/nums, "running_loss": running_loss/nums})

                if (epoch + 1) % 20 == 0:
                    torch.save(self.model.state_dict(), save_dir + "/final_transformer_exchang_enc_epoch-" + str(epoch + 1) + ".model")

                test_score = self.test(epoch)
                if test_score > best_score:
                    best_score = test_score
                    torch.save(self.model.state_dict(), save_dir + "/final_transformer_exchang_enc_epoch-best" + ".model")
                    print("Save for the best model")

    if (args.ssl_method == 'our_method') or (args.ssl_method == 'ablation_simple_3d'):
        def test(self, epoch):
            self.synth_views_model.eval()
            self.invar_view_model.eval()
            # self.invar_view_model_dual.eval()


            epoch_loss, ce_loss, smooth_loss = 0.0, 0.0, 0.0
            nums = 0
            with torch.no_grad():
                running_loss = 0.0
                running_gt_loss = 0.0
                running_loss_invar = 0.0
                for i, item in enumerate(test_loader):
                    nums += 1
                    samples_src = item[0].to(device)
                    samples_tgt = item[1].to(device)
                    count = item[2].to(device)
                    labels = item[3].to(device)
                    src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                    src_mask = src_mask.to(device)
                    src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)
                    src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)
                    cat_sync_vid = torch.cat([samples_src, samples_tgt], dim=2)               # shape: (4096,)
                    synth_rep = self.synth_views_model(cat_sync_vid) 

                    view_invar_rep_ego = self.invar_view_model(samples_src) # shape: (4096,)
                    view_invar_rep_ego += samples_src

                    view_invar_rep_exo = self.invar_view_model(samples_tgt) # shape: (4096,)
                    view_invar_rep_exo +=  samples_tgt

                    gt_loss = self.criterion_mse(samples_src, samples_tgt)
                    running_gt_loss += gt_loss.item()
                    loss_recon = self.criterion_mse(view_invar_rep_ego, synth_rep) + self.criterion_mse(view_invar_rep_exo, synth_rep)
                    loss_invar = self.criterion_l1(view_invar_rep_ego, view_invar_rep_exo)

                    running_loss_invar += loss_invar.item()
                    running_loss += loss_recon.item()

                pr_str = "***[epoch %d]***: test_running_loss_invar = %.3f, test_running_loss_mse = %.3f, gt_loss = %.3f, running_gt_loss = %.3f" % (epoch + 1, running_loss_invar / nums, running_loss / nums, gt_loss/nums, running_gt_loss/nums)
                print(pr_str)
                
            self.synth_views_model.train()
            self.invar_view_model.train()
            # self.invar_view_model_dual.train()
            return loss_recon
    elif args.ssl_method == 'sync_kd':
        def test(self, epoch):
            self.model.eval()
            epoch_loss, ce_loss, smooth_loss = 0.0, 0.0, 0.0
            nums = 0
            with torch.no_grad():
                running_loss = 0.0
                running_gt_loss = 0.0
                for i, item in enumerate(test_loader):
                    nums += 1
                    samples_src = item[0].to(device)
                    samples_tgt = item[1].to(device)
                    count = item[2].to(device)
                    labels = item[3].to(device)
                    src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
                    src_mask = src_mask.to(device)
                    src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

                    output = self.model(samples_tgt)
                    output = output + samples_tgt
                    gt_loss = self.criterion(samples_tgt, samples_src)
                    running_gt_loss += gt_loss.item()
                    loss = self.criterion(output, samples_src)
                    running_loss += loss.item()

                pr_str = "***[epoch %d]***: test_loss = %.3f, test_running_loss = %.3f, gt_loss = %.3f, running_gt_loss = %.3f" % (epoch + 1, loss / nums, running_loss / nums, gt_loss/nums, running_gt_loss/nums)
                print(pr_str)
                
            self.model.train()
            return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--feature_path', type=str, default='/mnt/data/zhanzhong/assembly/')
parser.add_argument('--teacher_feature_path', type=str, default='/mnt/data/zhanzhong/assembly/')
parser.add_argument('--dataset', default="assembly")
parser.add_argument('--split', default='train')  # or 'train_val'
parser.add_argument('--seed', default='42')
parser.add_argument('--test_aug', type=int, default=0)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--ssl_method', type=str, default='our_method')

args = parser.parse_args()

seed = int(args.seed)
set_seed(seed)
# VIEWS = ['C10095_rgb', 'C10115_rgb', 'C10118_rgb', 'C10119_rgb', 'C10379_rgb', 'C10390_rgb', 'C10395_rgb', 'C10404_rgb',
            #  'HMC_21176875_mono10bit', 'HMC_84346135_mono10bit', 'HMC_21176623_mono10bit', 'HMC_84347414_mono10bit',
            #  'HMC_21110305_mono10bit', 'HMC_84355350_mono10bit', 'HMC_21179183_mono10bit', 'HMC_84358933_mono10bit']

VIEWS = ['C10118_rgb']

config = dotdict(
    epochs=500,
    dataset=args.dataset,
    feature_size=2048,
    gamma=0.5,
    step_size=100,
    split=args.split)

if args.dataset == "assembly":
    config.chunk_size = 20
    config.max_frames_per_video = 1200
    config.learning_rate = 1e-5
    config.weight_decay = 1e-4
    config.batch_size = 8
    config.num_class = 202
    config.back_gd = []
    config.ensem_weights = [1, 1, 1, 1, 1, 1]
    if args.action == 'predict':
        if int(args.test_aug):
            config.chunk_size = list(range(10, 31, 7))
            config.weights = np.ones(len(config.chunk_size))
        else:
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
teacher_features_path = args.teacher_feature_path

config.features_path = features_path
config.teacher_features_path = teacher_features_path
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

