# IA025 - Introdução ao Aprendizado Profundo - Notas e exercícios

* [Ementa do curso](ementa.md)
* [Material de cursos passados](https://colab.research.google.com/github/robertoalotufo/rnap/blob/master/PyTorch/0_index.ipynb)

## Aula 1 - 17/03/2022
* [Exercício de programação Python, NumPy, PyTorch](Exercicios/Aula%201%20-%20Entrega%20-%20Exerc%C3%ADcios%20Introdut%C3%B3rios.ipynb)
* Leitura de artigo: [LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." nature 521.7553 (2015): 436-444.](https://s3.us-east-2.amazonaws.com/hkg-website-assets/static/pages/files/DeepLearning.pdf)
  * [Artigo anotado](Artigos/DeepLearning.pdf)
  * [Resumo do artigo](Resumos/Resumo-DeepLearning.pdf)
* Atividade adicional: 
  * [Exercício pré-curso Python - NumPy](Exercicios/ExercicioIntrodutorio.ipynb)
* Materiais complementares:
  * Video - [Jensen Huang — NVIDIA's CEO on the Next Generation of AI and MLOps](https://www.youtube.com/watch?v=kcI3OwQsBJQ)
  * Video - [Future Computers Will Be Radically Different](https://www.youtube.com/watch?v=GVsUOuSjvcg) - Analog computers and Artificial Inteligence
  * Artigo - [The Hardware Lottery](https://arxiv.org/abs/2009.06489) - o hardware que temos atualmente é o que dita quais algoritmos serão os que terão melhor desempenho
  * Postagem - [Deep Neural Nets: 33 years ago and 33 years from now](http://karpathy.github.io/2022/03/14/lecun1989/) - Andrej Karpathy - o que mudou no treinamento de redes neurais de 1989 até hoje

## Aula 2 - 24/03/2022
* [Exercício - Regressão_Linear e Grid de custos](Exercicios/Aula2-Regressão_Linear.ipynb)
* Leitura/resumo
  * Artigo: [Deep Neural Nets: 33 years ago and 33 years from now](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
    * [Artigo anotado](Artigos/lecun-89e.pdf)
  * Blog: [Deep Neural Nets: 33 years ago and 33 years from now](http://karpathy.github.io/2022/03/14/lecun1989/)
    * [Post anotado](Artigos/Deep%20Neural%20Nets%2033%20years%20ago%20and%2033%20years%20from%20now.pdf)
  * [Resumo](Resumos/Resumo%20-%20LeCun%20Backpropagation.pdf)
* Materiais complementares:
  * Vídeo - [Curso do Andrej Karpathy de Redes Neurais - aula de backpropagation](https://www.youtube.com/watch?v=gYpoJMlgyXA)
  * Artigo (Forbes) - [A Wave Of Billion-Dollar Language AI Startups Is Coming](https://www.forbes.com/sites/robtoews/2022/03/27/a-wave-of-billion-dollar-language-ai-startups-is-coming/) - explosão de startups voltadas para NLP+DL
  * Vídeo - Dicas de python: [25 nooby Python habits you need to ditch](https://www.youtube.com/watch?v=qUeud6DvOWI)
  * Artigo - Estudo sobre diferentes arquiteturas do Transformer - [Do Transformer Modifications Transfer Across Implementations and Applications?](https://arxiv.org/pdf/2102.11972.pdf?ref=https://githubhelp.com)

## Aula 3 - 31/03/2022
* [Exercício - Backpropagation](Exercicios/Aula3-BackPropagation.ipynb)
* Notas de aula: [Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition. Backpropagation Intuitions](https://cs231n.github.io/optimization-2/)
  * [Nota de aula anotada](Artigos/Notas%20de%20aula%20de%20Stanford%20CS231n%20-%20Backpropagation%20Intuitions.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20Notas%20de%20aula%20Stanford%20CS231n%20-%20Backpropagation%20Intuitions.pdf)
* Materiais complementares:
  * Vídeo - [Could a purely self-supervised Foundation Model achieve grounded language understanding?](https://www.youtube.com/watch?v=Tp412ab3kHQ)
  * Novo modelo de linguagem da Google: [Pathways Language Model - PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)
  * Artigo - Historia do aprendizado de máquina com o foco no debate computação vs "engenhosidade humana": [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

## Aula 4 - 07/04/2022
* [Exercício - DataLoader MNIST_Softmax Loss](Exercicios/Aula4-Regressao_Softmax_MNIST_SGD_minibatches.ipynb)
* Artigo: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  * [Artigo anotado](Artigos/NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper%20-%20anotado.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20ImageNet%20Classification%20with%20Deep%20Convolutional%20Neural%20Networks.pdf)
  * Vídeo: [revisão do artigo](https://youtu.be/Nq3auVtvd9Q)

## Aula 5 - 28/04/2022
* [Exercício - Implementação de convolução](Exercicios/Aula5-MNIST_Convolucional.ipynb)
* Artigo: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  * [Artigo anotado](Artigos/Deep%20Residual%20Learning%20for%20Image%20Recognition.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.pdf)

## Aula 6 - 05/05/2022
* [Exercício - Implementação de bloco residual e dropout](Exercicios/Aula_6_Exercício_ResNet_Dropout.ipynb)
* Artigo: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)
  * [Artigo anotado](Artigos/Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf)
  * [Resumo do artigo](Resumos/Resumo%20–%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf)

## Aula 7 - 12/05/2022
* [Exercício - Modelo de Linguagem (Bengio 2003) - MLP + Embeddings](Exercicios/Aula_7_LanguageModelBengio_Perplexity.ipynb)
* Artigo: [A Neural Probabilistic Language Model](https://arxiv.org/pdf/2103.00020.pdf)
  * [Artigo anotado](Artigos/A%20Neural%20Probabilistic%20Language%20Model.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20A%20Neural%20Probabilistic%20Language%20Model.pdf)

## Aula 8 - 19/05/2022
* [Exercício - Implementação de modelo de linguagem com auto-atenção](Exercicios/Aula_8_SelfAttention.ipynb)
* Artigo: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  * [Artigo anotado](Artigos/Attention%20Is%20All%20You%20Need.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20Attention%20is%20all%20you%20need.pdf)

## Aula 9 - 26/05/2022
## Aula 10 - 02/06/2022
## Aula 11 - 09/06/2022
## Aula 12 - 23/06/2022
## Aula 13 - 30/06/2022
## Aula 14 - 07/07/2022
## Aula 15 - 14/07/2022



