# Java - PyTorch Demo (Deep Java Learning)

The Deep Java Library (DJL) is a high-level, engine-agnostic framework
for deep learning in Java. It provides APIs to train and deploy models
without requiring deep knowledge of specific deep learning engines.

In this demo we show an introductory example of using a custom model
(cruise-drink-package) to inference (predict) based upon a given input.

# Training a Custom Model

To build our own model we use Jupyter notebook.

In the jupyter folder you will find 2 ipynb files.

CruiseNotebook.ipynb: Trains a drink package purchase prediction model
using an input csv file.

ConvertToOnnx.ipynb: Converts a joblib file into onnx model format.

## Data Format

``` text
Age: Number > 0
Gender: 0 (female), 1 (male)
Itinerary: SKU code for ship route. 0 -> 2 (three different routes).
Loyalty Level: No Level (0), Bronze (1), Silver(2), Gold(3).
Drink Package: No Package, Soda & Juices, Coffee/Tea/Hot Chocolate, Alcohol, Premium
```

To help illustrate trends we’re going to assume route 0 is a northern
itinerary (preference for hot drinks), and route 2 is tropical
(refreshment).

## Prediction Purpose

The model seeks to aid marketers in targeting sales to a specific
passenger based upon cruiser trends. Given known passenger information
(Age, Gender, Itinerary, Loyalty Level) which Drink Package upsell is
most likely to be sold if offered.

ex: What drink package would a 21-year-old Male with Silver loyalty on
Itinerary 0 or 2 likely purchase?

## joblib, Onnx and dot files

We have provided model files, and a dot file for the training set.

The dot file will requite a graphviz package for viewing in your IDE.

# DLJ Project Build

``` text
mvn clean install
```

# Run

``` text
mvn exec:java -Dexec.mainClass="com.savoir.java.pytorch.demo.OnnxDemo"
```
