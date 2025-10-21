import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

cluster_palette = [
    "#1f77b4",
    "#ff7f0e",
    "#279e68",
    "#d62728",
    "#aa40fc",
    "#aa40fc",
    "#e377c2",
    "#b5bd61",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#dbdb8d",
    "#9edae5",
    "#ad494a",
    "#8c6d31",
    "#b4d2b1",
    "#568f8b",
    "#1d4a60",
    "#cd7e59",
    "#ddb247",
    "#d15252",
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#ef476f",
    "#ffd166",
    "#06d6a0",
    "#118ab2",
    "#073b4c",
    "#fbf8cc",
    "#fde4cf",
    "#ffcfd2",
    "#f1c0e8",
    "#cfbaf0",
    "#a3c4f3",
    "#90dbf4",
    "#8eecf5",
    "#8359A3",
    "#5e503f",
    "#33CC99",
    "#F2C649",
    "#B94E48",
    "#0095B7",
    "#FF681F",
    "#e0aaff",
    "#FED85D",
    "#0a0908",
    "#C32148",
    "#98f5e1",
    "#000000",
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
    "#6A3A4C",
    "#00B7FF",
    "#004DFF",
    "#00FFFF",
    "#826400",
    "#580041",
    "#FF00FF",
    "#00FF00",
    "#C500FF",
    "#B4FFD7",
    "#FFCA00",
    "#969600",
    "#B4A2FF",
    "#C20078",
    "#0000C1",
    "#FF8B00",
    "#FFC8FF",
    "#666666",
    "#FF0000",
    "#CCCCCC",
    "#009E8F",
    "#D7A870",
    "#8200FF",
    "#960000",
    "#BBFF00",
    "#FFFF00",
    "#006F00",
]


def plot_embeddings_umap(ce, X, y=None, figname="umap.png", save_embeddings=False):
    embs = ce.get_embeddings_true(X)
    embs = embs.cpu().numpy()
    if save_embeddings:
        embs_name = "embeddings".join(figname.rsplit("UMAP.png", 1))
        np.save(embs_name, embs)

    for neighs in [20, 50, 100]:
        reducer = umap.UMAP(
            n_components=2, metric="cosine", random_state=42, n_neighbors=neighs
        )
        embeddings_2d = reducer.fit_transform(embs)
        df = pd.DataFrame(
            {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": y}
        )

        # Create the plot
        plt.figure(figsize=(10, 8))
        if y is not None:
            # palette = sns.color_palette("viridis", len(set(y)))=
            sns.scatterplot(
                x="x",
                y="y",
                hue="label",
                data=df,
                palette=cluster_palette,
                alpha=0.8,
                s=10,
            )

            # Calculate mean positions for each label to place the text
            for label in df["label"].unique():
                label_data = df[df["label"] == label]
                mean_x = label_data["x"].mean()
                mean_y = label_data["y"].mean()

                # Add text annotation for each label at the mean position
                plt.text(
                    mean_x,
                    mean_y,
                    str(label),
                    fontsize=10,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                )
        else:
            sns.scatterplot(x="x", y="y", data=df, color="#3f5b9a", alpha=0.8, s=10)
        plt.title("UMAP Plot of Embeddings")
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.legend(title="Cell types", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save the plot to a file
        plt.savefig(str(neighs) + figname, bbox_inches="tight")
        plt.close()
