# These imports bring in logical constructs and modeling tools from the lnn
# library. Each imported function or class has a specific role, such as
# creating propositions, logical relationships, or handling training.
from lnn import (Propositions, And, Implies, Iff, Fact, Model, Or, Loss,
                  Direction, Predicates, Variables, Not, World)

# Constants - Knowledge Setup
# This line defines five propositions, each representing a unique logical
# statement in the model.
A, B, C, D, E = Propositions("A", "B", "C", "D", "E")
# Defines a logical implication where proposition A implies proposition B.
IMPLIES = Implies(A, B)
AND = And(C, D)
IFF = Iff(AND, E)
# Combines the IMPLIES and IFF logical relationships into a single compound
# logical sentence using an AND operation.
SENTENCE = And(IMPLIES, IFF)

# Functions
# Defines a helper function to format and return the state of a given logical
# node, including its truth values rounded to one decimal place.
def query_state(node):
    round_off = lambda my_list: [float(f"{_:.1f}") for _ in my_list]
    return f"{node.state().name}: {tuple(round_off(node.get_data().tolist()))}"

# Data - Add the truth values of each node
# Sets the truth value of the SENTENCE node to TRUE, indicating it is a known
# fact in this context.
SENTENCE.add_data(Fact.TRUE)
A.add_data(0.9)  # Assigns a probabilistic truth value of 0.9 to proposition A.
B.add_data(Fact.FALSE)  # Assigns a FALSE value to proposition B.
E.add_data((0.2, 0.8))  # Assigns a range of truth values to proposition E.
SENTENCE.print()  # Prints the state of the SENTENCE node.
A.print()  # Prints the state of proposition A.
B.print()  # Prints the state of proposition B.
C.print()  # Prints the state of proposition C.
D.print()  # Prints the state of proposition D.
E.print()  # Prints the state of proposition E.

# Model Setup - Add knowledge
# Initializes a new logical neural network model to contain and process the
# knowledge defined in the script.
model = Model()
# Adds the SENTENCE logical relationship to the model's knowledge base for
# reasoning and inference.
model.add_knowledge(SENTENCE)
model.add_knowledge(IFF)  # Adds the IFF logical relationship to the model.
model.add_knowledge(AND)  # Adds the AND logical relationship to the model.
# Adds the IMPLIES logical relationship to the model.
model.add_knowledge(IMPLIES)
model.add_knowledge(A)  # Adds proposition A to the model.
model.add_knowledge(B)  # Adds proposition B to the model.
model.add_knowledge(C)  # Adds proposition C to the model.
model.add_knowledge(D)  # Adds proposition D to the model.
model.add_knowledge(E)  # Adds proposition E to the model.

# Reasoning
# Triggers the reasoning process, allowing the model to infer the states of
# unknown propositions based on the knowledge provided.
model.infer()
model.print(params=True)  # Prints the model parameters after inference.

# Creates a logical OR relationship among propositions C, D, and E to evaluate
# their combined truth values.
query = Or(C, D, E)
query.upward()  # Propagates truth values upward through the OR relationship.
query_state(query)  # Prints the state of the query node.

# Training - Upward Direction
# Trains the model using an upward direction to adjust its parameters and
# resolve contradictions in the logical relationships.
model.train(direction=Direction.UPWARD, losses=[Loss.CONTRADICTION])
model.print(params=True)  # Prints the model parameters after training.

SENTENCE.print()  # Prints the state of the SENTENCE node.
AND.print()  # Prints the state of the AND node.
IFF.print()  # Prints the state of the IFF node.
IMPLIES.print()  # Prints the state of the IMPLIES node.
A.print()  # Prints the state of proposition A.
B.print()  # Prints the state of proposition B.
C.print()  # Prints the state of proposition C.
D.print()  # Prints the state of proposition D.
E.print()  # Prints the state of proposition E.

query.upward()  # Re-evaluates the OR relationship after training.
query_state(query)  # Prints the updated state of the query node.

# Training - Downward Direction
# Trains the model in the downward direction to adjust its parameters and
# resolve contradictions.
model.train(direction=Direction.DOWNWARD, losses=[Loss.CONTRADICTION])
# Prints the model parameters after downward training.
model.print(params=True)

SENTENCE.print()  # Prints the state of the SENTENCE node.
AND.print()  # Prints the state of the AND node.
IFF.print()  # Prints the state of the IFF node.
IMPLIES.print()  # Prints the state of the IMPLIES node.
A.print()  # Prints the state of proposition A.
B.print()  # Prints the state of proposition B.
C.print()  # Prints the state of proposition C.
D.print()  # Prints the state of proposition D.
E.print()  # Prints the state of proposition E.

model.print(params=True)  # Prints the final model parameters.

# Example Propositions
A, B = Propositions("A", "B")  # Defines two new propositions A and B.
IMPLIES = Implies(A, B)  # Creates a logical implication where A implies B.
A.add_data(0.9)  # Assigns a probabilistic truth value of 0.9 to proposition A.
B.add_data(Fact.FALSE)  # Assigns a FALSE value to proposition B.
A.print()  # Prints the state of proposition A.
B.print()  # Prints the state of proposition B.
IMPLIES.print()  # Prints the state of the IMPLIES node.

model = Model()  # Initializes a new model.
model.add_knowledge(IMPLIES)  # Adds the IMPLIES relationship to the model.
model.infer()  # Infers the state of the IMPLIES relationship.
IMPLIES.print()  # Prints the inferred state of the IMPLIES node.

# Initialize an empty model
model = Model()  # Creates an empty model for predicate-based reasoning.

# Predicates Setup
# Defines a binary predicate to represent friendship relationships between two
# entities.
Friends = Predicates('Friends', arity=2)
# Defines a binary predicate to represent 'likes' relationships between two
# entities.
Likes = Predicates('Likes', arity=2)
# Defines three variables for use in predicate logic.
x, y, u = Variables('x', 'y', 'u')

# Creates a rule stating that if x and y are friends and x likes u, then y also
# likes u.
Friends_like_Films = Implies(And(Friends(x, y), Likes(x, u)), Likes(y, u))
# Groups the logical rule into a list of formulae.
formulae = [Friends_like_Films]
# Adds the formulae to the model as axioms.
model.add_knowledge(*formulae, world=World.AXIOM)

# Data - Add to the model
# Adds specific instances of the 'Friends' and 'Likes' relationships to the
# model as data, including both TRUE and FALSE values.
model.add_data({
    Friends: {
        ('a', 'b'): Fact.TRUE,  # Specifies that 'a' and 'b' are friends.
        ('a', 'c'): Fact.TRUE,  # Specifies that 'a' and 'c' are friends.
        ('b', 'c'): Fact.TRUE,  # Specifies that 'b' and 'c' are friends.
        ('c', 'd'): Fact.TRUE,  # Specifies that 'c' and 'd' are friends.
    },
    Likes: {
        ('a', 'j'): Fact.TRUE,  # Specifies that 'a' likes 'j'.
        ('a', 'l'): Fact.TRUE,  # Specifies that 'a' likes 'l'.
        ('b', 'l'): Fact.TRUE,  # Specifies that 'b' likes 'l'.
        ('c', 'l'): Fact.TRUE,  # Specifies that 'c' likes 'l'.
        ('d', 'j'): Fact.TRUE,  # Specifies that 'd' likes 'j'.
        ('c', 'j'): Fact.FALSE,  # Specifies that 'c' does not like 'j'.
    },
})
model.print()  # Prints the model's knowledge base.

# Inferencing
model.infer()  # Infers new relationships based on the provided data and rules.
# Prints the model's parameters and inferred knowledge.
model.print(params=True)

# Training - Loss Contradiction
# Trains the model to minimize contradictions in its knowledge base.
model.train(losses=Loss.CONTRADICTION)
model.print(params=True)  # Prints the model parameters after training.
