![Unicamp](logo_unicamp.png)

# IA025 - Introdução ao Aprendizado Profundo

## Materiais principais 
* [Ementa do curso](ementa.md)
* [Material de cursos passados](https://colab.research.google.com/github/robertoalotufo/rnap/blob/master/PyTorch/0_index.ipynb)
* [Apresentações das aulas](Presentation_IA025A_2022S1.pdf)
* [Lista dos exercícios entregues](Exercicios/README.md)
* [Lista dos artigos lidos, anotados e resumidos](Resumos/README.md)
* Projeto final da disciplina:
  * [Relatório](https://github.com/marcusborela/exqa-complearning/blob/main/Relatório_Final_Projeto_exqa-complearning.pdf)
  * [Apresentação](https://github.com/marcusborela/exqa-complearning/blob/main/docs/presentations/CompLearningExQA_final_presentation.pdf)
  * [Quadro Miro](https://miro.com/app/board/uXjVOr04EAw=/?share_link_id=606867964752)
  * [Github do projeto](https://github.com/marcusborela/exqa-complearning)

## Roteiro das aulas e atividades

### Aula 1 - 17/03/2022
* Introdução ao curso
  * Regras
  * Motivação Deep Learning (apresentação)
* Assuntos
  * Numpy: 
    * Operações matriciais
    * imagens
    * broadcast
    * redução de eixo
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

### Aula 2 - 24/03/2022
* Assuntos
  * Introdução à redes neurais (1 camada apenas)
  * Notebooks de Regressão Linear
  * Notebooks de Regressão Logística
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

### Aula 3 - 31/03/2022
* Assuntos
  * Grafo Computacional
  * Backpropagation
* [Exercício - Backpropagation](Exercicios/Aula3-BackPropagation.ipynb)
* Notas de aula: [Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition. Backpropagation Intuitions](https://cs231n.github.io/optimization-2/)
  * [Nota de aula anotada](Artigos/Notas%20de%20aula%20de%20Stanford%20CS231n%20-%20Backpropagation%20Intuitions.pdf)
  * [Resumo da nota](Resumos/Resumo%20-%20Notas%20de%20aula%20Stanford%20CS231n%20-%20Backpropagation%20Intuitions.pdf)
* Materiais complementares:
  * Vídeo - [Could a purely self-supervised Foundation Model achieve grounded language understanding?](https://www.youtube.com/watch?v=Tp412ab3kHQ)
  * Novo modelo de linguagem da Google: [Pathways Language Model - PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)
  * Artigo - Historia do aprendizado de máquina com o foco no debate computação vs "engenhosidade humana": [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

### Aula 4 - 07/04/2022
* Assuntos
  * Regressão Logística (Classificador Softmax)
  * Dataset e Dataloader do Pytorch
  * Dados de Treinamento, Validação e Testes
* [Exercício - DataLoader MNIST_Softmax Loss](Exercicios/Aula4-Regressao_Softmax_MNIST_SGD_minibatches.ipynb)
* Artigo: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  * [Artigo anotado](Artigos/NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper%20-%20anotado.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20ImageNet%20Classification%20with%20Deep%20Convolutional%20Neural%20Networks.pdf)
  * Vídeo: [revisão do artigo](https://youtu.be/Nq3auVtvd9Q)
* Materiais complementares:
  * Vídeo (Chris Potts / Stanford) - Modelos treinados apenas com auto-supervisão são capazes de "entender" o mundo?](https://www.youtube.com/watch?v=Tp412ab3kHQ)
  * Caderno - [Exemplo de como utilizar a GPU](https://colab.research.google.com/drive/1UF6TZn5005Cx1xy1RBh4ZHcqF-Z4EWkD?usp=sharing)

### Aula 5 - 28/04/2022
* Assuntos
  * Ativações
  * redes convolucionais
* [Exercício - Implementação de convolução](Exercicios/Aula5-MNIST_Convolucional.ipynb)
* Artigo: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  * [Artigo anotado](Artigos/Deep%20Residual%20Learning%20for%20Image%20Recognition.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.pdf)
* Materiais complementares:
  * Vídeos para melhor entendimento do artigo da semana:
    * [Deep Residual Learning for Image Recognition (Paper Explained)](https://www.youtube.com/watch?v=GWt6Fu05voI)
    * [C4W2L03 Resnets](https://www.youtube.com/watch?v=ZILIbUvp5lk&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=14)
    * [C4W2L04 Why ResNets Work](https://www.youtube.com/watch?v=RYth6EbBUqM&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=15)

### Aula 6 - 05/05/2022
* Assuntos
  * Transfer Learning
  * Aumento de dados
  * BatchNorm/LayerNorm
  * Conexões residuais
  * Dropout
  * Overfit em um batch
* [Exercício - Implementação de bloco residual e dropout](Exercicios/Aula_6_Exercício_ResNet_Dropout.ipynb)
* Artigo: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)
  * [Artigo anotado](Artigos/Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf)
  * [Resumo do artigo](Resumos/Resumo%20–%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf)
* Materiais complementares:
  * Video - [Palestra do Prof. Lotufo sobre os avanços recentes em Processamento de Linguagem Natural](https://www.youtube.com/watch?v=hEmpPfkP3TE&t=1s)
  * Video (Google) - [Modelo PALM de 540B de parametros capaz de explicar piadas](https://www.youtube.com/watch?v=nP-nMZpLM1A&t=7240s)
  * Artigos - [Trabalhos da Timnit sobre vieses sociais](https://scholar.google.com/citations?user=lemnAcwAAAAJ
  * Vídeo - [Documentário da Netflix sobre bias na IA - Coded Bias](https://www.uol.com.br/tilt/noticias/redacao/2021/04/10/coded-bias-da-netflix-prova-como-a-tecnologia-e-racista-e-viola-direitos.htm) 

### Aula 7 - 12/05/2022
* Assuntos
  * Modelos de Linguagem
  * Perplexidade
* [Exercício - Modelo de Linguagem (Bengio 2003) - MLP + Embeddings](Exercicios/Aula_7_LanguageModelBengio_Perplexity.ipynb)
* Artigo: [A Neural Probabilistic Language Model](https://arxiv.org/pdf/2103.00020.pdf)
  * [Artigo anotado](Artigos/A%20Neural%20Probabilistic%20Language%20Model.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20A%20Neural%20Probabilistic%20Language%20Model.pdf)
* Materiais complementares:
  * Artigo - [Batchnorm](https://arxiv.org/pdf/1502.03167.pdf) - seção 3.2: "Note that, since we normalize Wu_b, the bias can be ignored.

### Aula 8 - 19/05/2022
* Assuntos
  * Transformer
  * Auto-atenção
  * Normalização dos escores de atenção
* [Exercício - Implementação de modelo de linguagem com auto-atenção](Exercicios/Aula_8_SelfAttention.ipynb)
* Artigo: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  * [Artigo anotado](Artigos/Attention%20Is%20All%20You%20Need.pdf)
  * [Resumo do artigo](Resumos/Resumo%20-%20Attention%20is%20all%20you%20need.pdf)
* Materiais complementares:
  * Artigo - [Librispeech - dataset bastante popular para avaliar sistemas de reconhecimento da fala](https://paperswithcode.com/sota/speech-recognition-on-librispeech-test-clean)
  * Artigo - [wav2vec 2.0 - transformer perto do estado da arte para esta reconhecimento da fala](https://arxiv.org/pdf/2006.11477.pdf)
  * Artigo - [Como diferenças na implementação do transformer geram resultados diferentes](https://arxiv.org/abs/2102.11972)

### Aula 9 - 26/05/2022
* Assuntos
  * Transformer (Decoder-only)
  * Atenção causal / auto-regressão
  * Padding
  * Multi-head
* [Exercício - Atenção multi-cabeça, auto-atenção causal e padding](Exercicios/Aula_9_MultiheadAttention_Causal.ipynb)
* Artigo: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
  * [Cópia do artigo](Artigos/Language%20Models%20are%20Few-Shot%20Learners.pdf)
  * [Resumo do artigo](Resumos/Resumo%20–%20Language%20Models%20are%20Few-Shot%20Learners.pdf)
* Materiais complementares:
  * Artigos com ilustrações da auto-atenção: 
    * [Transformer from scratch using pytorch](https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch/notebook)
    * [Transformers Explained Visually (Part 3): Multi-head Attention, deep dive](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853#:~:text=Decoder%20Self%2DAttention&text=of%20each%20word.-,This%20is%20fed%20to%20all%20three%20parameters%2C%20Query%2C%20Key%2C,for%20each%20word%20as%20well)
  * Demo - [BertViz - Visualize Attention in NLP Models](https://github.com/jessevig/bertviz)
  * Artigo - [BERTology](https://arxiv.org/pdf/2002.12327.pdf)

### Aula 10 - 02/06/2022
* Assuntos:
  * Revisão Transformers
  * Discussão dos projetos
* Exercício repetido: [Atenção multi-cabeça, auto-atenção causal e padding](Exercicios/Aula_9_MultiheadAttention_Causal.ipynb)
* Artigo: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
  * [Artigo anotado](Artigos/Scaling%20Laws%20for%20Neural%20Language%20Models.pdf)
  * [Resumo do artigo](Resumos/Resumo%20–%20Scaling%20Laws%20for%20Neural%20Language%20Models.pdf)
* Materiais complementares:
  * Caderno - [Utilização do GPT-J-6B (versão open-source do GPT-3) no colab pro](https://colab.research.google.com/drive/1tclcJWQkfOY-midXBb7SaAdrG5yMl-T-?usp=sharing)
  * Vídeo: [Yannick Kilchner explicando o GPT-3](https://www.youtube.com/watch?v=SY5PvZrJhLE)

### Aula 11 - 09/06/2022
* Assuntos:
  * Image Captioning
  * Discussão dos projetos
* Projeto escolhido (Leo/Borela): OpenQA com Few-shot
* Materiais complementares:
  * Vídeo: [Entrevista com o Geoffrey Hinton (padrinho do deep learning)](https://www.youtube.com/watch?v=2EDP4v-9TUA)
  * Modelo: [Geração de imagens a partir de textos (text-to-image) - DALLE-mini](https://huggingface.co/dalle-mini/dalle-mini)
  * Vídeo: [Jacob Devlin apresentando o BERT](https://www.youtube.com/watch?v=knTc-NQSjKA)
  * Vídeo: [Visual Transformer (ViT) do Yannic](https://www.youtube.com/watch?v=TrdevFK_am4)
  * Caderno: [Como rodar o GPT-Neo de 20B parametros no colab](https://colab.research.google.com/drive/1GYRvsOhOlJqOabz6X9oEpzCtqoNM0roW?usp=sharing#scrollTo=IcAeIg0IKrMX)

### Aula 12 - 23/06/2022
* Apresentação dos planos dos projetos
  * [Apresentação inicial](docs/presentations/CompLearningExQA-Presentation-20220623.pdf)
  * [Github do projeto](https://github.com/marcusborela/exqa-complearning)

### Aula 13 - 30/06/2022
* Apresentação dos projetos - andamento parcial
  * [Apresentação 30/06/2022](docs/presentations/CompLearningExQA-Presentation-20220630.pdf)
  * [Github do projeto](https://github.com/marcusborela/exqa-complearning)

### Aula 14 - 07/07/2022
* Apresentação dos projetos - andamento parcial
  * [Apresentação 07/07/2022](docs/presentations/CompLearningExQA-Presentation-20220707.pdf)
  * [Github do projeto](https://github.com/marcusborela/exqa-complearning)

### Aula 15 - 14/07/2022
* Apresentação final dos projetos
  * [Relatório](Relatório_Final_Projeto_exqa-complearning.pdf)
  * [Apresentação](docs/presentations/CompLearningExQA_final_presentation.pdf)
  * [Quadro Miro](https://miro.com/app/board/uXjVOr04EAw=/?share_link_id=606867964752)
  * [Github do projeto](https://github.com/marcusborela/exqa-complearning)
