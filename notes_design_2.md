# Design Smell 2

## Building Inefficient Data Pipelines

### Questions to Answer:

1. How can building inefficient data pipelines make Convolutional Neural Nets less effective?
2. What are some trademarks or indicators that my data pipeline is inefficient?
3. How can I ensure my data pipeline is optimized for efficiency (and maximizing my model's usefulness)?

### Resources:

*As any of these don't prove useful, remove them from this doc*

1. [How to Improve Deep Learning Performance](https://machinelearningmastery.com/improve-deep-learning-performance/)

2. [Integrating Convolutional Neural Networks into Real-World Applications: Common Challenges and Ways to Overcome Them](https://perfectial.com/blog/convolutional-neural-networks/)

3. [An overview of deep learning in medical imaging focusing on MRI](https://www.sciencedirect.com/science/article/pii/S0939388918301181)

4. [Building a Data Pipeline for Deep Learning](https://www.lenovonetapp.com/pdf/wp-7299.pdf)

5. [CNN-Peaks: ChIP-Seq peak detection pipeline using convolutional neural networks that imitate human visual inspection](https://www.nature.com/articles/s41598-020-64655-4)


### Notes from reading:

1. How to Improve Deep Learning Performance:

- Improving performance with data:

  - **Get more data**
    - Neural networks perform better the more data they're trained on. Can you reasonably get more data? Then try to get more data.

  - **Invent more data**
    - Don't have enough data, and can't get more data? Invent more training examples.
      - (copied from source):
        If your data are vectors of numbers, create randomly modified versions of existing vectors.
        If your data are images, create randomly modified versions of existing images.
        If your data are text, you get the idea…
      - Often called "data augmentation" or "data generation"

  - **Rescale your data**
    - A traditional rule of thumb when working with neural networks is to scale your data to the bounds of your activation function.
      - (copied from source):
        If you are using sigmoid activation functions, rescale your data to values between 0-and-1. If you’re using the Hyperbolic Tangent (tanh), rescale to values between -1 and 1.

        This applies to inputs (x) and outputs (y). For example, if you have a sigmoid on the output layer to predict binary values, normalize your y values to be binary. If you are using softmax, you can still get benefit from normalizing your y values.
    - His (author's) suggestion is also to create a few different versions of your training dataset like so:
      - Normalized to 0-1
      - Rescaled to -1 - 1
      - Standardized
    - Evaluate the performance of your model on each variation, and pick one accordingly. Then, he says, "double down"
    - If you change your activation function, change this approach accordingly
    - You want to avoid "big values" accumulating in your network (there are other methods for this as well)

  - **Transform your data**
    - More work than rescaling, but related
    - Must really get to know your data, visualize it, look for outliers
    - Basically, what he says here, is to experiment and try things with your data, with the goal of exposing your network to the "structure of the problem"
    - Some methods, copied from source:

      Guesstimate the univariate distribution of each column.

        Does a column look like a skewed Gaussian, consider adjusting the skew with a Box-Cox transform.
        Does a column look like an exponential distribution, consider a log transform.
        Does a column look like it has some features, but they are being clobbered by something obvious, try squaring, or square-rooting.
        Can you make a feature discrete or binned in some way to better emphasize some feature.

      Lean on your intuition. Try things.

        Can you pre-process data with a projection method like PCA?
        Can you aggregate multiple attributes into a single value?
        Can you expose some interesting aspect of the problem with a new boolean flag?
        Can you explore temporal or other structure in some other way?

    - In short, really get to know your data so you can understand how to work with it, how to omit noise before you get to the feature engineering step. Play with features, play with transforms, and see what works and what doesn't.

  - **Feature Selection**:
    - While neural nets are generally pretty good at figuring out on their own what are and aren't predictive features, helping omit some useless noise or zero-in on predictive features can help both your speed and performance.
      - We don't want to be using data, weights, and training cycles on data that isn't needed to make good predictions
      - Can you remove attributes from your data?
      - There are a lot of feature selection methods, try some (or all) and figure out what works for your problem
      - A few ideas the author proposes:
        - Maybe you can do as well or better with fewer features (this will make training and inferencing faster)
        - Maybe all the feature selection methods you try get rid of the same subset of features - this is good, you've found a consensus on useless features in your data
        - Maybe a selected subset gives you some more ideas on further feature engineering you can perform

  - **Reframe your problem**:
    - Evaluate alternate framings and ways of approaching your problem that might better expose the structure of your problem to your net

2. Integrating Convolutional Neural Networks into Real-World Applications: Common Challenges and Ways to Overcome Them

  - Batching:
    - Something to add to article: "are you leveraging batching?"
      - This enables the weights to update multiple times per epoch by training your CNN on batches of inputs rather than the entire dataset each epoch
      - Enables faster convergence and less chance of overfitting
      - Greater operational efficiency
  - What to do if there's a lack of data?
    - Brings up transfer learning - which is really valuable to talk about
    - Basically, use a model that's already been trained on a general problem
      - Then, starting with higher layers and working your way back, fine-tune the layers (perform backpropagation on your dataset) and determine where you should stop working backwards
      - The intuition is that for especially computer vision tasks, CNNs learn really general features first (i.e: "diagonal patters" or something of the sort) and then can learn more specific problems at higher layers, but you don't need to train from scratch necessarily
      - This alleviates needing so much data, and optimizes the training resources you use.

3. An overview of deep learning in medical imaging focusing on MRI

  - Building blocks of CNNs:
    - You could theoretically use a typicall feed-forward neural network on image tasks, but it's inefficient to have one connection of each neuron in a layer to each neuron of every other layer.
      - To specifically work with images, we need some "domain knowledge" - i.e: knowledge of the structure of images, to very specifically prune our network and obtain better performance
    - A CNN is a particular kind of ANN that is aimed at preserving spatial relationships in the data, while retaining very few connections between hidden layers
    - Input to a CNN is arranged in a grid structure (images are, implicitly) and then fed through layers which preserve these relationships, each layer operation operating on a small portion of the previous layer
      - A CNN has multiple layers of *convolutions* and *activations*, often interspersed with *pooling* layers and is trained using backpropagation and gradient descent (as do standard neural networks)

4. Building a Data Pipeline for Deep Learning

  - Stages in the data pipeline for deep learning:
    1. Ingest (gather data/source data)
    2. Data prep (cleaning, preprocessing)
    3. Training (fit the model)
    4. Validation (test the model - evaluate your results)
      - Several cycles of data prep, training and validation are often required to find the model, parameters, and preprocessing methods that yield the best or desired results.
    5. Deploy
    6. Archive
  - Steps to ensure an efficient data pipeline:
    - Know your business needs (OR your use case/problem case)
    - Know your data needs
      - Where does your data come from now?
        - (my additions) do you know how much you'll need?
        - Do you know what it needs to look like? What labels?
      - Do you have data on deck?
        - If not, what infrastructure is going to be required to get it?
        - If yes, do you need any external data to augment this data? (i.e: weather information, social media analytics, etc.)
      - For each source, do you have a right to that data?
        - You must be sure that you're keeping privacy and security in mind, and that you're not violating any regulations or guidelines.
        - You must be sure you can access this data in a consistent format in the necessary timeframe.
  - Keep in mind the three V's:
    - *Volume*: as a rule, the greater volume of data the better the performance of the deep learning system.
    - *Variety*: Variety means having diverse attributes and features in the dataset. The greater the variety, the more accurately a deep learning model can generalize.
    - *Veracity*: Correct labeling of data is key


**Takeaways**:

Common issues:

- Not enough data
- Not properly labeled or robust data
- Misunderstanding the problem/taking shortcuts
  - Misunderstanding use case, trying to tackle a really complex use case before knowing what you need for your pipeline
  - Can't source data ethically, can't source data quickly enough, etc.
  - Inefficient use of data
  - Not having enough variety within data to allow net to learn patterns
  - Alternatively, leaving an excess of noise or useless attributes
    - Solution: feature selection tools
- Over-budget/overuse of compute resources, financial resources, time
  - Solution: if you have little data or less compute resources, try a transfer learning approach
  - Always start small, don't just dive into deep learning for no reason (is this even a data pipeline thing?)


### Outline:

1. How can building inefficient data pipelines make Convolutional Neural Nets/ANNs less effective?
2. What are some trademarks or indicators that my data pipeline is inefficient?
3. How can I ensure my data pipeline is optimized for efficiency (and maximizing my model's usefulness)?

I. What is an inefficient data pipeline?

  - A pipeline which is under-utilizing or under-optimizing your data, or in which you're overextending compute, financial and time resources unnecessarily.
  - A pipeline which isn't getting you the results you need despite using state-of-the-art ANNs

II. How can I tell that the way I'm dealing with my data may be the issue?

  - Tells:
    - Too little data with ANNs: it doesn't matter how many optimizations you make, you're not improving your success metric (accuracy, loss, etc.)
      - Diverging metrics could be an indication of overfitting due to too few training examples (this can also be an issue with the way you're building your model)
    - Overuse of resources/overextending budget without results
      - Indications that you might be misunderstanding you problem, lacking the right data, or attempting to go "too big too fast"
    - ANN reaches good metrics, but performs poorly on test data or "in the wild"
      - Another sign of overfitting that wasn't caught during training, a lack of variety in your data or use of out-of-domain data

III. What can I do to ensure my data pipeline is optimized for the best results possible?

  - Understand your problem
    - Take a step back. Have you performed a reasonable amount of research? Do you have the domain-specific knowledge that you need?
    - What is the desired outcome?
  - Understand your data and your data needs:
    - What data do you already have? (data assets)
      - Is it representative of the problem as it stands today? Is it properly labeled? Does it contain enough information? Is the information relevant?
    - What data do you need? How can you obtain it?
    - Are there external pieces of information your model might need that it doesn't have?
  - Get more data
    - Data collection, data augmentation where appropriate
  - Can't get more data or have a lack of compute resources?
    - Good case for transfer learning (depending on use case)

IV. Wrap up
