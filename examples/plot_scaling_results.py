import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

lattice = [5, 6, 7, 8, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 10]
mem = [
    "6414496",
    "16720588",
    "39187912",
    "0",
    "16626564",
    "47880360",
    "119723700",
    "266947288",
    "509222172",
    "6301356",
    "11634504",
    "32786012",
    "44419572",
    "44503264",
    "32774060",
    "61046932",
    "49393744",
    "122933164",
    "20503724",
    "29476348",
    "47336060",
    "123092028",
]

name = [
    "cpu_5",
    "cpu_6",
    "cpu_7",
    "cpu_8",
    "torch_5",
    "torch_6",
    "torch_7",
    "torch_8",
    "torch_9",
    "ignore",
    "ignore",
    "ignore",
    "ignore",
    "ignore",
    "ignore",
    "ignore",
    "ignore",
    "ignore",
    "jax_6_1n4d",
    "jax_7_1n4d",
    "jax_8_1n4d",
    "jax_10_2n8d",
]

wall_time = [
    "00:41:14",
    "02:30:04",
    "07:16:58",
    "timeout",
    "00:03:32",
    "00:19:10",
    "00:45:57",
    "02:56:00",
    "killed",
    "00:02:03",
    "00:01:07",
    "00:05:53",
    "00:07:44",
    "00:06:49",
    "00:06:28",
    "00:06:50",
    "00:10:20",
    "00:44:38",
    "00:01:24",
    "00:02:50",
    "00:04:52",
    "00:38:46",
]

energy = [
    "-94.938368",
    "-94.959899",
    "-94.9695380",
    "-94.938467",
    "-94.959978",
    "-94.969621",
    "-94.972020",
    "0",
    "-94.959977",
    "-94.959959",
    "-94.969553",
    "-94.972007",
    "-94.972021",
    "-94.96960",
    "0",
    "-94.972021",
    "-94.972021",
    "-94.97263",
    "-94.959970",
    "-94.969595",
    "-94.972021",
    "-94.972619",
]


def process_lattice_data(lattice, name, wall_time, energy_list):

    df = pl.DataFrame({
                          "lattice": lattice,
                          "name": name,
                          "time": wall_time,
                          "mem": mem,
                          "energy": energy
                      }).filter(
                                (pl.col("name") != "ignore") &
                                (pl.col("name") == "cpu_8") &
                                (pl.col("name") == "torch_9")
                            )

    print(df)
    df.write_csv("lattice_results.csv")

    plot_df = df
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=plot_df, x="lattice", y="energy", hue="framework", marker="o")
    plt.title("Energy Convergence across Lattices")
    plt.xlabel("Lattice Size")
    plt.ylabel("Energy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("energy_convergence.png")

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=plot_df, x="lattice", y="wall_time_sec", hue="framework", marker="s"
    )
    plt.yscale("log")
    plt.title("Lattice Size vs Wall Time (Log Scale)")
    plt.xlabel("Lattice Size")
    plt.ylabel("Wall Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("wall_time_vs_lattice.png")

    print("DataFrame Head:")
    print(df.head())
    return df


df = process_lattice_data(lattice, name, wall_time, energy)
