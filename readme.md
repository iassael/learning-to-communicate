
# Learning to Communicate with Deep Multi-Agent Reinforcement Learning

Jakob N. Foerster, Yannis M. Assael, Nando de Freitas, Shimon Whiteson

We consider the problem of multiple agents sensing and acting in environments with the goal of maximising their shared utility. In these environments, agents must learn communication protocols in order to share information that is needed to solve the tasks. By embracing deep neural networks, we are able to demonstrate end-to-end learning of protocols in complex environments inspired by communication riddles and multi-agent computer vision problems with partial observability. We propose two approaches for learning in these domains: Reinforced Inter-Agent Learning (RIAL) and Differentiable Inter-Agent Learning (DIAL). The former uses deep Q-learning, while the latter exploits the fact that, during learning, agents can backpropagate error derivatives through (noisy) communication channels. Hence, this approach uses centralised learning but decentralised execution. Our experiments introduce new environments for studying the learning of communication protocols and present a set of engineering innovations that are essential for success in these domains.

## Execution
```
$ # Requirements: nvidia-docker
$ # Build docker instance
$ ./build.sh
$ # Run docker instance
$ ./run.sh
$ # Run experiment e.g.
$ ./run_switch_3-dial.sh
```

## Bibtex

    @article{foerster2016learning,
        title={Learning to Communicate with Deep Multi-Agent Reinforcement Learning},
        author={Foerster, Jakob N and Assael, Yannis M and de Freitas, Nando and Whiteson, Shimon},
        journal={arXiv preprint arXiv:1605.06676},
        year={2016}
    }

## License

Code licensed under the Apache License v2.0