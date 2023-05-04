import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib import cm


class SelectFromCollection:
    def __init__(self, ax, collection, alpha_other=0.1):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        lineprops = {"color": "black", "alpha": 0.8}

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect, props=lineprops)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def plotData(df, gridsize, aggregation_type):
    fig, ax = plt.subplots()

    pts = ax.hexbin(
        x=df.x,
        y=df.y,
        C=df["FAX001FT144t.PV"],
        gridsize=gridsize,
        cmap=cm.get_cmap("seismic"),
        edgecolors="black",
        reduce_C_function=aggregation_type,
    )
    plt.colorbar(pts, ax=ax)
    selector = SelectFromCollection(ax, pts)

    def accept(event):
        if event.key == "enter":
            points_df = pd.DataFrame(selector.xys[selector.ind])
            points_df.rename(columns={0: "X", 1: "Y"}, inplace=True)
            points_df.to_json("xy_indices.json")
            plt.scatter(points_df.X, points_df.Y, s=20)
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()
            return points_df

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")
    plot = plt.show()
    return plot


if __name__ == "__main__":
    df = pd.read_parquet("data/latents.parquet", engine="fastparquet")

    plotData(df, 50, np.mean)
