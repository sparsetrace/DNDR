from sec_torus_demo import run_demo

out = run_demo()
sec = out["sec"]
result = out["result"]
print("DMAP epsilon:", sec.epsilon)
print("Leading SEC eigenvalues:", sec.sec_evals_[:8])
print("Tangent singular values:", result.singular_values[:5])
print("Principal angles vs true torus tangent plane (deg):", out["principal_angle_degrees"])
sec.plot(result, filename="/mnt/data/sec_torus_plot.html")
print("Saved plot to /mnt/data/sec_torus_plot.html")
