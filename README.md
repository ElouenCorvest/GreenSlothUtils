# GreenSlothUtils

This package includes all the commands needed to simplify the work on the GreenSloth Project.

## How to Create a model for GreenSloth (Using CLI)

In thsi section you are getting a step-by-step guide on how the easiest way is to prepare a model for GreenSloth.

### 0. All the commands

Using `GreenSlothUtils` enables an easier way to contribute to GreenSloth. While there are many different functions and upsides this custom package gives, the most important ones are summarized in a CLI interface. These commands initializes your work for contributing a model to GreenSloth.

Once you have the ``GreenSlothUtils` package in your environment (we recommend using [`pixi`](https://pixi.sh/latest/)) you can enter the following in your Terminal. Any Terminal should work.

```bash
GreenSloth-init --help
```

This command shows you all the availeble options in the CLI interface. All the commands are also explained here and given an order to fully encapsulate the way, we best believe, you should summarise for GreenSloth.

### 1. Create the Directory

Every model should in a seperate directory, for ease of distinction. by using the following command you can do just that:

```bash
GreenSloth-init initialize <model-name> -p=<path>
```

The argument `<model-name>` is mandatory and represents the name of the directory. We recommend using the name of your model, as it makes it much easier. Additionally, all created files that use the model name will also be using this given argument. So choose wisely!

The option `-p=<path>`, is optional and lets you give the command a specific path where the directory should be created. If this is ommited, the path where the Terminal points to will be taken.

### 2. Create your model using `MxLPy`

This step is sadly not easily replaced by `GreenSlothUtils`. You will have to create your own working model and we recommend doing so with `MxLPy` if applicable. To see how to use that package please go its [docs](https://github.com/Computational-Biology-Aachen/MxlPy). Inside your newly created model directory, you will see the model directory, that contains several python-scripts. These scripts are there to guide you through the model-creating process and help you to keep a tidy representation fo your model.

The `__init__.py` file is the main file of your model. It collects all the other files and actually combines them to create your model, which should be using the name of the parent directory. The only things you should add to this file, are the variables and the parameters.

The `basic_funcs.py` file is empty, as it is a file for you to put common functions used by your model. The rest of the model files shoudl automatically import all the functions you define here.

The `derived_quantities.py` and the `rates.py` file should include, as their appropriate names show, the derived quantitites and the rates of your model. Please include any addition to these files inside the function, which name's you should not change! These two functions are directly called by the `__init__.py` file to construct the rest of your model.

At the end, you should be able to import the model folder and use the `<model_name>()` function (where you replace the `<model_name>` with the appropriate name) to constuct your entire model in any other python-script. This function is also in some of the commands used by `GreenSlothUtils` 