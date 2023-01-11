Day 1
Tried to use the Neural Magic Model "neuralmagic/oBERT-12-upstream-pruned-unstructured-97". The macro and micro f1 scores were much smaller at the beginning of the model; the initial step did not increase much. However, it did outperform in the same epoch by .159 difference in the f1 score.
Modification of the code was more significant was able to add errors in my program to move to the CPU if there was an error in my program
import gc
'''
Try and Catch block when training of the model use more memory than the GPU, it will produce an error.

1. Check the Amount of GPU memory used
2. Move the model to CPU
3. Call the garbage collector
4. Free the GPU memory in the cache
5. Check the amount of GPU memory used to see if it is freed
'''
def check_gpu_memory():
    print(torch.cuda.memory_allocated()/1e9)
    return torch.cuda.memory_allocated()/1e9
try:
    trainer.train()
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("CUDA out of memory")
        print("Let's free some GPU memory and re-allocate")
        check_gpu_memory()
        ## Move the model to CPU
        model.to("cpu")
        gc.collect()
        ## Free the GPU memory
        torch.cuda.empty_cache()
        check_gpu_memory()
    else:
        raise e
Able to check if there was a number of support my model can support in my model
from transformers import Trainer, TrainingArguments
def is_on_colab():
    if 'google.colab' in sys.modules:
        return True
    return False

training_args_fine_tune = TrainingArguments(
    output_dir  = "./multi-label-class-classification-on-github-issues" ,
    num_train_epochs = 15,
    learning_rate = 3e-5,
    per_device_train_batch_size = 64 ,
    evaluation_strategy = "epoch" ,
    save_strategy="epoch"  ,
    load_best_model_at_end=True,
    metric_for_best_model='micro f1',
    save_total_limit=1,
    log_level='error',
    push_to_hub = True  if is_on_colab else False ,
    )
if torch.cuda.is_available():
    ## check if the Cuda GPU can bfloat16
    if torch.cuda.is_bf16_supported():
        print("Cuda GPU can support bfloat16")
        training_args_fine_tune.fp16 = True
    else:
        print("Cuda GPU cannot support bfloat16 so instead we will use float16 ")
        training_args_fine_tune.fp16 = True
Day 2
Add Augmentation to the dataset
I was hoping the diffence will be hueg immediatly instead . it was a slow process to get the ballin running. At the same time, in no difference in the performance of the model. The stop back had stop running. It did not seed an difference of the performance of the seed of the model . Adding the Augmentation finall broke the celinof the fperofrmance oc the model at the macro level of the model performance. The performance has increase to my likely. So I confident adding more Augementation which are common to dataset that is online
KeyboardAug
Substitute word according to spelling mistake dictionary
Split one word to two words randomly
Use TF-IDF to find out how word should be augmented
Leverage word2vec, GloVe or fasttext embeddings to apply augmentation
I added weight label which is for the loss function. Hoping to get the error to increase the amount of time.
Baseline for adding Synonymous Augmentation
    aug = naw.SynonymAug(
        aug_src = "wordnet",
        aug_min = 0 ,
        aug_max = 100 ,
        lang = "eng" ,
        aug_p = .3
    )
With the same epoch the difference in the performance of the model is difference
{
'eval_loss': 0.10408153384923935,
'eval_micro f1': 0.6589633542423242,
'eval_macro f1': 0.07207471928913042,
'eval_runtime': 2.3849,
'eval_samples_per_second': 326.215,
'eval_steps_per_second': 41.091,
'epoch': 16.0
}
Compare to the old method which was
Loss: 0.1391
Micro f1: 0.5005
Macro f1: 0.0340
For the macro performance has double. At the same the micro performance has double . At the same time, the training has increase. The question. The dataset has increase.
Screenshot Capture - 2023-01-10 - 19-11-13.png

The time did not increase much
