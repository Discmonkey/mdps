import mdptoolbox.example
P, R = mdptoolbox.example.forest(S=10, r1=2, r2=3)

print P, R
# vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
# vi.verbose = True
# vi.run()
# print vi.policy
