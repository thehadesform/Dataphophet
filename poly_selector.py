import json
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
import matplotlib.pyplot as plt
import pandas as pd

POLYGON_JSON_FILENAME = "polygon_points.json"


class SelectFromCollection:

    def __init__(self, ax, collection, alpha_other=0.1):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        lineprops = {"color": "green", "alpha": 0.8}

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect, props=lineprops)
        self.vertiecs = []
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        self.vertices = path.vertices

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def save_points_json(polygon_points):
    with open(POLYGON_JSON_FILENAME, "w") as outfile:
        json.dump(polygon_points.tolist(), outfile)


def scatterplot_polygon_select(df, store_json=True):
    _, ax = plt.subplots()
    pts = ax.scatter(x=df.x, y=df.y)
    selector = SelectFromCollection(ax, pts)
    plt.show()
    selector.disconnect()
    if store_json:
        save_points_json(selector.xys[selector.ind])
    return df.iloc[selector.ind, :]


if __name__ == "__main__":


    df = pd.read_parquet("data/latents.parquet")
    sub_df = scatterplot_polygon_select(df)
