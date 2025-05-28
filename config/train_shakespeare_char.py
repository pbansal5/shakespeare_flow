# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
from datetime import datetime

timestep = datetime.now().strftime("%m-%d_%H-%M-%S")
out_dir = 'out-shakespeare-char-embed256_flow%s'%timestep
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 2
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = out_dir
out_dir = 'runs/%s'%out_dir

generative_type = 'flow'
if (generative_type == 'ar'):
    train_embeddings = True
elif (generative_type == 'flow'):
    train_embeddings = False # this defaults to loading embeddings from "embeddings_file"
    # embeddings_file should be set as the ckpt path of the statedict you want embeddings of
    # embeddings_file = '/u/pbansal/nanoGPT/runs/out-shakespeare-char-embed65_05-23_15-11-23/ckpt.pt' 
    embeddings_file = '/u/pbansal/nanoGPT/runs/out-shakespeare-char-embed256_05-25_11-42-01/ckpt.pt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1 # for AR training we keep it at 1. for flow we take 4
batch_size = 1024 # for AR training we keep it at 64. 
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6 # for AR model this is 6. 
n_head = 6 # for AR model this is 6. 
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 20_000 # for AR training we keep it at 10K. For Flow training increase to 100K. 
warmup_iters = 100 # not super necessary potentially

lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small


# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
