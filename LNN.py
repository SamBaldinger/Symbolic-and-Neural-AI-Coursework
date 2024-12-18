# Imports
from lnn import (Propositions, And, Implies, Iff, Fact, Model, Or, Loss,
                  Direction, Predicates, Variables, Not, World)

# Constants - Knowledge Setup
# Set up the propositions A, B, C, D, E.
A, B, C, D, E = Propositions("A", "B", "C", "D", "E")
IMPLIES = Implies(A, B)
AND = And(C, D)
IFF = Iff(AND, E)
SENTENCE = And(IMPLIES, IFF)

# Functions
def query_state(node):
    round_off = lambda my_list: [float(f"{_:.1f}") for _ in my_list]
    return f"{node.state().name}: {tuple(round_off(node.get_data().tolist()))}"

# Data - Add the truth values of each node
SENTENCE.add_data(Fact.TRUE)
A.add_data(0.9)
B.add_data(Fact.FALSE)
E.add_data((0.2, 0.8))
SENTENCE.print()
A.print()
B.print()
C.print()
D.print()
E.print()

# Model Setup - Add knowledge
model = Model()
model.add_knowledge(SENTENCE)
model.add_knowledge(IFF)
model.add_knowledge(AND)
model.add_knowledge(IMPLIES)
model.add_knowledge(A)
model.add_knowledge(B)
model.add_knowledge(C)
model.add_knowledge(D)
model.add_knowledge(E)

# Reasoning
model.infer()
model.print(params=True)

query = Or(C, D, E)
query.upward()
query_state(query)

# Training - Upward Direction
model.train(direction=Direction.UPWARD, losses=[Loss.CONTRADICTION])
model.print(params=True)

SENTENCE.print()
AND.print()
IFF.print()
IMPLIES.print()
A.print()
B.print()
C.print()
D.print()
E.print()

query.upward()
query_state(query)

# Training - Downward Direction
model.train(direction=Direction.DOWNWARD, losses=[Loss.CONTRADICTION])
model.print(params=True)

SENTENCE.print()
AND.print()
IFF.print()
IMPLIES.print()
A.print()
B.print()
C.print()
D.print()
E.print()

model.print(params=True)

# Example Propositions
A, B = Propositions("A", "B")
IMPLIES = Implies(A, B)
A.add_data(0.9)
B.add_data(Fact.FALSE)
A.print()
B.print()
IMPLIES.print()

model = Model()
model.add_knowledge(IMPLIES)
model.infer()
IMPLIES.print()

# Initialize an empty model
model = Model()

# Predicates Setup
Friends = Predicates('Friends', arity=2)
Likes = Predicates('Likes', arity=2)
x, y, u = Variables('x', 'y', 'u')

Friends_like_Films = Implies(And(Friends(x, y), Likes(x, u)), Likes(y, u))
formulae = [Friends_like_Films]
model.add_knowledge(*formulae, world=World.AXIOM)

# Data - Add to the model
model.add_data({
    Friends: {
        ('a', 'b'): Fact.TRUE,
        ('a', 'c'): Fact.TRUE,
        ('b', 'c'): Fact.TRUE,
        ('c', 'd'): Fact.TRUE,
    },
    Likes: {
        ('a', 'j'): Fact.TRUE,
        ('a', 'l'): Fact.TRUE,
        ('b', 'l'): Fact.TRUE,
        ('c', 'l'): Fact.TRUE,
        ('d', 'j'): Fact.TRUE,
        ('c', 'j'): Fact.FALSE,
    },
})
model.print()

# Inferencing
model.infer()
model.print(params=True)

# Training - Loss Contradiction
model.train(losses=Loss.CONTRADICTION)
model.print(params=True)
