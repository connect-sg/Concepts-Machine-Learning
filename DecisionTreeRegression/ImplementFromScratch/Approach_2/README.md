Steps - 
1. Calculate information gain with each possible split
2. Divide set with that feature and value that gives the most IG
3. Divide tree and do the same for all created branches
4. ...until a stopping critera is reached


What we need to decide on - 
Split feature
Split point

Information gain => IG = E(parent) - [weight average].E(Childrent)
Entropy => E = -SUMMATION(p(X).log2(p(X)))
p(X) = #x/n