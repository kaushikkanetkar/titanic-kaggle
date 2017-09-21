## Implementation of the Titanic Problem using Tensorflow
## Classifiers used are Linear, DNN and the combined Deep-and-Wide model

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as Learn

import sklearn
import sys
import argparse
from sklearn.cross_validation import train_test_split


CATEGORICAL_COLUMNS = ["Child", "Sex", "Cab", "NameT", "AgeKnown", "Embarked", "Young", "Family", "MaleBadTicket"]
CONTINUOUS_COLUMNS = ["Age", "Fare", "Pclass", "NameLength"]
tf.logging.set_verbosity(tf.logging.ERROR)

def build_estimator(model_dir, classifier):

  # Categorical columns
  sex = tf.contrib.layers.sparse_column_with_keys(column_name="Sex",
                                                     keys=["female", "male"])
  family = tf.contrib.layers.sparse_column_with_keys(column_name="Family",
                                                       keys=["Large", "Nuclear", "Solo"])
  child = tf.contrib.layers.sparse_column_with_keys(column_name="Child",
                                                       keys=["0", "1"])
  ageknown = tf.contrib.layers.sparse_column_with_keys(column_name="AgeKnown",
                                                       keys=["0", "1"])
  embarked = tf.contrib.layers.sparse_column_with_keys(column_name="Embarked",
                                                       keys=["C", "S", "Q"])
  young = tf.contrib.layers.sparse_column_with_keys(column_name="Young",
                                                       keys=["0", "1"])
  malebadticket = tf.contrib.layers.sparse_column_with_keys(column_name="MaleBadTicket",
                                                       keys=["0", "1"])
  cab = tf.contrib.layers.sparse_column_with_hash_bucket(
      "Cab", hash_bucket_size=10)
  namet = tf.contrib.layers.sparse_column_with_hash_bucket(
      "NameT", hash_bucket_size=20)

  # Continuous columns
  age = tf.contrib.layers.real_valued_column("Age")
  namelength = tf.contrib.layers.real_valued_column("NameLength")
  fare = tf.contrib.layers.real_valued_column("Fare")
  p_class = tf.contrib.layers.real_valued_column("Pclass")

  # Transformations.
  fare_buckets = tf.contrib.layers.bucketized_column(fare,
  						     boundaries=[
						        5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550
						     ])
  age_buckets = tf.contrib.layers.bucketized_column(age,
                                                    boundaries=[
                                                        5, 18, 25, 30, 35, 40, 45, 50, 55, 65
                                                    ])
  pclass_buckets = tf.contrib.layers.bucketized_column(p_class,
                                                    boundaries=[1, 2, 3])

   # Wide columns and deep columns.
  wide_columns = [sex, cab, namet, child, ageknown, embarked, young, family,
                  tf.contrib.layers.crossed_column(
                      [age_buckets, sex],
                      hash_bucket_size=int(1e3)),
		  tf.contrib.layers.crossed_column(
		      [pclass_buckets, sex],
                      hash_bucket_size=int(1e3)),
		  tf.contrib.layers.crossed_column(
		      [fare_buckets, pclass_buckets],
                      hash_bucket_size=int(1e3)),
		  tf.contrib.layers.crossed_column(
		      [embarked, pclass_buckets],
                      hash_bucket_size=int(1e3)),
		  tf.contrib.layers.crossed_column(
		      [embarked, sex],
                      hash_bucket_size=int(1e3))]


  deep_columns = [
      namelength,
      fare,
      p_class,
      tf.contrib.layers.embedding_column(sex, dimension=8),
      tf.contrib.layers.embedding_column(child, dimension=8),
      tf.contrib.layers.embedding_column(family, dimension=8),
      tf.contrib.layers.embedding_column(cab, dimension=8),
      tf.contrib.layers.embedding_column(namet, dimension=8),
      tf.contrib.layers.embedding_column(ageknown, dimension=8),
      tf.contrib.layers.embedding_column(embarked, dimension=8),
      tf.contrib.layers.embedding_column(young, dimension=8),
      tf.contrib.layers.embedding_column(malebadticket, dimension=8)
  ]

  if classifier == "deep":
    return Learn.DNNClassifier(model_dir=model_dir,
                               feature_columns=deep_columns,
                               hidden_units=[32, 16],
                               optimizer=tf.train.ProximalAdagradOptimizer(
                               learning_rate=0.1,
                               l2_regularization_strength=0.001))
  elif classifier == "wide":
    return Learn.LinearClassifier(
            feature_columns=wide_columns,
            optimizer=tf.train.FtrlOptimizer(
                    learning_rate=5,
                    l1_regularization_strength=1000.0,
                    l2_regularization_strength=1000.0),
                    model_dir=model_dir)
  else:
    return Learn.DNNLinearCombinedClassifier(
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[32, 16],
            model_dir=model_dir,   
	    linear_optimizer=tf.train.FtrlOptimizer(
	                        learning_rate=10,
				l1_regularization_strength=100.0,
				l2_regularization_strength=100.0),
            dnn_optimizer=tf.train.ProximalAdagradOptimizer(
	                            learning_rate=0.1,
                                    l2_regularization_strength=0.001))

def input_fn(df, y, train=False):

  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  categorical_cols = {k: tf.SparseTensor(
    indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values,
    dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}

  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)

  if train:
    label = tf.constant(y.values)
    return feature_cols, label
  else:
    return feature_cols

def derive_feature_from(derived):

  derived['AgeKnown'] = np.where(derived['Age'].isnull() == False, 0, 1).astype(str)
  derived["Age"] = derived["Age"].fillna(derived["Age"].median()).astype(int)
  derived['Child'] = np.where(derived['Age']< 9, 1, 0).astype(str)
  derived['FamSize'] = derived['SibSp'] + derived['Parch'] + 1
  derived['Family'] = np.where((derived['FamSize'] == 1), 'Solo', np.where((derived['FamSize'] <= 4), 'Nuclear', 'Large'))
  derived['Alone'] = np.where((derived['SibSp'] + derived['Parch'] == 0), 1, 0).astype(str)
  derived["Embarked"] = derived["Embarked"].fillna("S")
  derived["Cabin"] = derived["Cabin"].fillna('U')
  derived['Cab'] = derived['Cabin'].map(lambda c : c[0])
  derived['NameLength'] = derived['Name'].apply(lambda x: len(x)).astype(int)
  derived['NameT'] = derived['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
  derived['Young'] = np.where(((derived['Age']<=30) | (derived['NameT'].isin(['Master','Miss','Mlle']))), 1, 0).astype(str)
  derived['Ttype'] = derived['Ticket'].str[0]
  derived['BadTicket'] = np.where(derived['Ttype'].isin(['3','4','5','6','7','8','A','L','W',' ']), 1, 0)
  derived['MaleBadTicket'] = np.where(((derived['BadTicket']) | (derived['Sex'] == 'male')), 1, 0).astype(str)

  return derived

def train_and_eval(train_steps, classifier):

  df_train = pd.read_csv(tf.gfile.Open("train.csv"), skipinitialspace=True)
  df_test = pd.read_csv(tf.gfile.Open("test.csv"), skipinitialspace=True)

  df_train = derive_feature_from(df_train)
  df_test = derive_feature_from(df_test)

  y = df_train['Survived']
  X = df_train[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS]

  X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=123)

  model_dir = ""

  m = build_estimator(model_dir, classifier=classifier)
  m.fit(input_fn=lambda: input_fn(X_train, y_train, True), steps=train_steps)

  results_test = m.evaluate(input_fn=lambda: input_fn(X_test, y_test, True), steps=1)
  print("\nTest results\n")
  for key in sorted(results_test):
      print("%s: %s" % (key, results_test[key]))

  # Output data
  y_pred = df_test['Survived']
  X_pred = df_test[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS]
  prediction = m.predict(input_fn=lambda: input_fn(X_pred, y_pred))
  pid = pd.DataFrame(data=df_test["PassengerId"], columns=['PassengerId'])
  result = pd.DataFrame(data=prediction, columns=['Survived'])
  result = pd.concat([pid, result], axis=1) 
  result.to_csv("output.csv", index=False) 

  #Accuracy on training data
  results = m.evaluate(input_fn=lambda: input_fn(X_train, y_train, True), steps=1)
  
  print("\nTraining results\n")
  for key in sorted(results):
      print("%s: %s" % (key, results[key]))

def main(_):
  train_and_eval(FLAGS.train_steps, FLAGS.classifier)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_steps",
      type=int,
      default=12000,
      help="Training steps"
  )
  parser.add_argument(
      "--classifier",
      type=str,
      default='deep_wide',
      help="Type"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
