from typing import List, Optional
import typer
from zeroshot.models import HuggingFaceZeroShotClassifier, USE4ZeroShotClassifier

app = typer.Typer()


@app.command()
def gender(names: List[str],
           use4: Optional[bool] = False,
           ):

    clf = USE4ZeroShotClassifier() if use4 else HuggingFaceZeroShotClassifier()
    res = clf.predict_list(inputs=names, options=['Female', 'Male'])

    for name_ix, name in enumerate(names):
        msg = typer.style(f'* Prediction for {name} is {res[name_ix]}', fg=typer.colors.GREEN, bold=True)
        typer.echo(msg)
