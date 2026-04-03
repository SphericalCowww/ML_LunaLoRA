# Multi-LoRAs System with LLM Core and Continuous Timestamped Training
Seems like global LLM models are no longer limited by model size, but by their structure. The goal here is to combat catastrophic forgetting with a local agent performing continuous timestamped learning, and that "**Caching is so last century, real memories ought to be stored directly in NN.**"

The general idea is to keep the central LLM unchanged while surrounding it with a dozen LoRAs, each with an assigned weight and orthogonal to the others. The system is made to be enclosed in an agent that does real-time observation, i.e., a robot, and perform continuous training. Every short moment, each LoRA will be queried with "Summarize your observation and your last output, and what are the times?" followed by the time information and the real-time data from the agent. At the same time, the LLM will be queried with "Summarize your observation and the most recent user input," followed by the real-time data from the agent and the user input (separating the user input from other observations, leaving room for reinforcement learning). A machine then updates the relevant weights between the outputs of the LoRAs and the LLM and trains on the LoRAs with the learning rate scaled by the weights. While using most of the core/RAM for training, the other core/RAM is used to continue generating output from LoRAs that include commands for the agent, e.g., move or reply back in voices.

The idea is that once the LoRAs reach a threshold, say thousands, the multi-LoRA system will begin to exhibit emergent properties towards general AGI, such as long-term memory and planning.

## Concepts

### Real memory ought to be stored in the Neural Network

The idea basically originates from "You know, Neural Network weights can store everything we know of, why not use them to store memory instead of temporary KV-caching or external databases? After all, most human don't have photographic memory, but rather internalize their experiences." This is followed by a parallel speculation that a global entity will have a difficult time avoiding becoming just a giant dictionary; only local agents with real-time experience of their surroundings can develop self-awareness. 

Based on these two ideas, the proposal is to use the base Large Language Model (LLM) as a static core and Low-Rank Adaptations (LoRAs) as dynamic "satellites" on an enclosed local agent with real-time observation using its sensors or other inputs. The system treats fine-tuning as the primary method for long-term memory storage. Sort of treating LLM as a permanent brain and LoRAs as the active "hippocampus." 

Then, by assigning specific weights to each LoRA, the system can balance between specializing and diversifying its knowledge sets. The most relevant LoRA to the observation will receive the highest priority in both learning and acting on the observation. To determine the relevance of a LoRA to the observation, outputs from the LLM alone will be compared to the outputs of each LLM+LoRA to derive the weights. These weights should directly influence the learning rate of each LoRA and activation for inference.

Finally, to prevent multiple LoRAs from learning the same information or "collapsing" into a single state, this approach also requires enforcing mathematical orthogonality between them. 

### Real-time training demands timestamping your memory

While the agent takes real-time observations, if these observations are fed directly into the LoRAs, the LoRAs will likely have a difficult time learning about the time-ordering of their experience. To cheat a little bit, perhaps bake timestamps directly into the training data, enforcing the chronological sense of history directly into the system. The idea is to make it learn the flow of time, not just a list of facts.

The only way for the agent to have real-time memory is to have it train continuously. Of course, this is limited by the computational power, so let's say we redo the LoRAs training each minute. Fortunately, the LLM is just there, so we can make it summarize the cumulative observation before feeding the information to train the LoRAs. Assuming nothing much happened, we can probably cut that learning session and have a variable time interval for training.

In the end, the LLM is just a Recurrent Neural Network, so it should, in principle, create a feedback loop with the observation data merged in between. By analyzing user reactions against its own previous actions, the agent can perform "retrospection." However, without a reward/target, the training will likely go nowhere. Hence, the user input should be open for Reinforcement Learning (RL) potential.

Lastly, implementing continuous training while performing inference will also be a challenge, but it should be resolvable. 

### Goal and prospect 

I am only an AI hobbyist, unaware of most of the literature, which makes me bold enough to throw these long-shot claims.

However, the "by increasing Compute, Data Size, and Model Parameters, the loss drops follow a predictable power law." is probably reaching its architectural limit. New architecture is likely required for the future emergence of the state towards general AGI, such as long-term planning, deep memory, and a persistent "personality" that evolves every second. 

My belief is that, if the LunaLoRA system is sound, once the system hits a critical mass, e.g., 1000+ LoRAs, the next phase transition in the emergent AGI will occur. And the number of LoRAs will define the next power law.

## Project

Big claims with hardly any implementation, but as a basic plan:

A system with Qwen3 as the core and 3 LoRAs. The agent is limited to the computer observing the computer clock, GPU usage, and the local weather report. Need to figure out how to do basic weight + orthogonality.

## References:
- John Schulman in collaboration with others at Thinking Machines, LoRA Without Regret (2025) (<a href="https://thinkingmachines.ai/blog/lora/">Blog</a>)
- 51616, Text-to-LoRA (T2L): Instant Transformer Adaption (LoRA Reinforcement Learning) (2025) (<a href="https://github.com/SakanaAI/text-to-lora">GitHub</a>)
