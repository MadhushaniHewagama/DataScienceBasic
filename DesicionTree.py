#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Training Data set
training_data = [
    ['Green', 3, 'Mango'],
    ['Yellow', 3, 'Mango'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]


# In[7]:


# Column labels
# These are used only to print the tree
header = ["color","diameter","label"]


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

########
# Demo:
# unique_vals(training_data, 0)
# unique_vals(training_data, 1)
########

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {} # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

########
# Demo:
# class_counts(training_data)
#######

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)
#########
# Demo:
#is_numeric(5)
#is_numeric("Madhu")
#is_numeric(5.5)
#########


# In[34]:


class Question:
    """A Question is used to partition a dataset.
    
    This class just records a 'column number' (e.g., 0 for Color) and a 'column value' (e.g., Green). 
    The 'match' method is used to compare the feature value in an example to the feature value stored in the question. 
    See the demo below.
    """
    
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
    def __repr__(self):
        # This is just a helper method to print
        # the question in  a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

    def partition(rows, question):
        """Partitions a dataset.

        For each row in the dataset, check if it matches the question. If 
        so, add it to 'true rows', otherwise, add it to 'false rows'.

        """

        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
        #######
        # Demo:
        # Let's partition the training data based on whether rows areRed.
        # true_rows, false_rows = partition(training_data, Question(0,'Red'))
        # print("True set",str(true_rows))
        # print("False set",str(false_rows))
        #######

    def gini(rows):
        """Calculate the Gini Impurity for a list of rows."""

        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity

    def info_gain(left, right, current_uncertainty):
        """Information Gain.

        The uncertainty of the starting node, minus the weighted impurity of two child nodes.
        """        
        p = float(len(left)) / (len(left)+len(right))
        return current_uncertainty - p * gini(left) - (1-p) * gini(right)

        ########
        # Demo
        # Calculate the uncertainy of our training data.
        # current_uncertainty = gini(training_data)
        # 
        # How much information do we gain by partioning on 'Green'?
        # true_rows, false_rows = partition(training_data, Question(0, 'Green'))
        # info_gain(true_rows, false_rows, current_uncertainty)
        ########

    def find_best_split(rows):
        """FInd the best question to ask by iterating over every feature / value
           and calculating the information gain."""
        best_gain = 0 # keep track of the best information gain
        best_question = None # keep train of the feature / value that produced it
        current_uncertainty = gini(rows)
        n_features = len(rows[0]) - 1 # number of columns

        for col in range(n_features): # for each feature
            values = set([row[col] for row in rows]) # unique values in the column
            for val in values: # for ech value
                question = Question(col,val)

                #try splitting the dataset
                true_rows, false_rows = partition(rows, question)

                # skip this split if it doesn't divide the dataset
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = info_gain(true_rows, false_rows, current_uncertainty)

                # you actually can use '>' instead of '>=' here
                # but I wanted the tree to look a certain way for our toy dataset
                if gain >= best_gain:
                    best_gain, best_question = gain, question
        return best_gain, best_question


# In[35]:


class Leaf:
    """A Leaf node classifies data.
    
    This holds a dictionary of class (e.g., "Mango") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """
    
    def __init__(self, rows):
        self.predictions = class_counts(rows)


# In[36]:


class Decision_Node:
    """A Decision Node asks a question.
    
    This holds a reference to the question, and to the two child nodes.
    """
    
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        


# In[38]:


def build_tree(rows):
    """Builds the tree."""
    
    # Try partitioning the dataset n each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)
    
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)
    
    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)
    
    # Recursively build the true branch
    true_branch = build_tree(true_rows)
    
    # Recursively build the false branch
    false_branch = build_tree(false_rows)
    
    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # depending on the answer
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    """ World's most elegant tree printing function."""
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return
    
    # Print the question at this node
    print ( spacing + str(node.question))
    
    # call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    
    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    
    # Base case: we've reached a leaf
    if isinstance(node,Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """Print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl]/total * 100)) +"%"
    return probs

########
# Demo:
# Printing that a bit nicer
my_tree= build_tree(training_data)
print_leaf(classify(training_data[0],my_tree))


# In[ ]:




