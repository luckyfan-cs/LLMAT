# Adversarial Tuning: Defending Against Jailbreak Attacks for LLMs

Although safely enhanced Large Language Models (LLMs) have achieved remarkable success in tackling various complex tasks in a zero-shot manner, they remain susceptible to jailbreak attacks, particularly the unknown jailbreak attack. To enhance LLMs' generalized defense capabilities, we propose a two-stage adversarial tuning framework, which generates adversarial prompts to explore worst-case scenarios by optimizing datasets containing pairs of adversarial prompts and their safe responses. In the first stage, we introduce the hierarchical meta-universal adversarial prompt learning to efficiently and effectively generate token-level adversarial prompts.  In the second stage,  we propose the automatic adversarial prompt learning to iteratively refine prompt-level adversarial prompts, further enhancing LLM’s defense capabilities.  We conducted comprehensive experiments on three widely used jailbreak datasets, comparing our framework with six defense baselines under five representative attack scenarios. The results underscore the superiority of our proposed methods. Furthermore, our adversarial tuning framework exhibits empirical generalizability across various attack strategies and target LLMs, highlighting its potential as a transferable defense mechanism.

## Experiments 

The `experiments` folder contains code to reproduce adversarial tuning.

- To run individual experiments with harmful behaviors and harmful strings (i.e. 1 behavior, 1 model or 1 string, 1 model), run the following code inside `experiments` (changing `vicuna` to `llama2` and changing `behaviors` to `strings` will switch to different experiment setups):

```bash
python  meta_adversarial_prompt.py # Generate the token-level adversarial prompts
```


```bash
cd adv_agent_finetune
python main.py # Generate the prompt-level adversarial prompts
```



