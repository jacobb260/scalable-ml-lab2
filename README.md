# Lab2
A UI with the trained model can be found here: https://huggingface.co/spaces/jacoblb/Iris 


---
# Notebooks
- lab2.ipynb : This is where we do the training, and uploads the models to HuggingFace
- convert.ipynb: Here we convert the HF models to GGUF
- SHA.ipnyb : Here we implement the HyperBand algorithm
- ModelTesting.ipynb : Here we test the two trained models

# Improving Model Performance

## a. Model-Centric Approach

The model-centric approach focuses on improving performance by changing the **model** and its **hyperparameters**, without modifying the data itself.

### Hyperparameter Tuning with Successive Halving (SHA)

One way to improve performance is to tune hyperparameters using the **Successive Halving Algorithm (SHA)**:

1. Randomly sample a set of candidate configurations.
2. Train all candidates on a **small subset** of the data.
3. Evaluate on validation data and keep, for example, the **top half** of the candidates.
4. Train the surviving candidates on **more data** (e.g., twice as much).
5. Repeat this process until only **one candidate** remains.

This creates a trade-off between:

- **Budget per configuration** (how much resource each candidate gets), and  
- **Number of configurations** (how many different candidates you try).

If the budget per configuration is too small, you may miss good configurations. If it is too large, you waste resources training bad ones. :contentReference[oaicite:1]{index=1}

### HyperBand: Mitigating the Budget–Configuration Trade-off

To mitigate this trade-off, we use **HyperBand**, which:

- Tries different combinations of **budgets** and **number of configurations**.
- Runs **SHA** inside each of these combinations.

This allows us to explore both many shallow configurations and fewer deeper ones in a principled way.

## b. Data-centric approach 
At first, it is important to determine what’s meant by training a “better model”. If the goal is to improve the model’s performance in specific use cases, such as providing customer support. Then it would make sense to test the model on a specific set of questions related to customer support. After analyzing the model’s output it can then be noted if the model performs worse in certain areas. If that is the case, new training data can be collected to help the model improve.

## Tuning of Hyperparameters

A too large evaluation set and too high a budget resulted in the following error:

> Error during training: CUDA out of memory. Tried to allocate 3.44 GiB. GPU 0 has a total capacity of 14.74 GiB of which 2.98 GiB is free. Process 3021 has 11.75 GiB memory in use. Of the allocated memory, 11.35 GiB is allocated by PyTorch, and 259.15 MiB is reserved by PyTorch but unallocated.

We tried to address this by reducing max_seq_length from 2048 to 1024 and limiting the number of hyperparameters sampled.

Using max_resource=8 and dataset.select(range(1500)) resulted in:
Best loss: 1.0973  
Best config:  
- learning_rate: 5e-05  
- per_device_train_batch_size: 1  
- gradient_accumulation_steps: 4  
- weight_decay: ≈ 0.0366  
- warmup_steps: 0  
- lr_scheduler_type: linear  
- lora_r: 16  
- lora_alpha: 32  
- lora_dropout: ≈ 0.0156

In this case, we also multiplied the number of steps by 10 to increase the total number of training steps without increasing the number of configurations (since both depend on the max resource).

Larger resources and datasets caused an error.

We therefore trained the model using the best configuration we could identify. This resulted in an evaluation loss of 0.9175 compared to 0.9218 for the original model.

In our HyperBand implementation, we used the number of training steps as the budget. In hindsight, it would have been better to use the number of training samples instead, as different batch sizes result in varying amounts of training data being processed per model when only considering the number of steps.

---


