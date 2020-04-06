from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits=load_digits()

fig,axes = plt.subplots(2,5, figsize=(10,5),subplot_kw={"xticks":(),"yticks":()})
for ax,img in zip(axes.ravel(),digits.images):
    ax.imshow(img)
