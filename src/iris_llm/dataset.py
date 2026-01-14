from typing import List

from pydantic import BaseModel, Field
from sklearn.datasets import load_iris


class IrisSample(BaseModel):
    """Represents a single Iris flower measurement."""
    sepal_length: float = Field(..., alias="sepal length (cm)")
    sepal_width: float = Field(..., alias="sepal width (cm)")
    petal_length: float = Field(..., alias="petal length (cm)")
    petal_width: float = Field(..., alias="petal width (cm)")
    target_name: str | None = None

    def to_natural_language(self) -> str:
        """Converts the numerical data into a descriptive sentence for the LLM."""
        return (
            f"This flower has a sepal length of {self.sepal_length}cm, "
            f"a sepal width of {self.sepal_width}cm, "
            f"a petal length of {self.petal_length}cm, "
            f"and a petal width of {self.petal_width}cm."
        )

def get_iris_data(sample: int = None) -> List[IrisSample]:
    """
    Loads the Iris dataset and returns a list of validated IrisSample objects.
    """
    iris = load_iris(as_frame=True)
    df = iris.frame

    if sample is not None:
        df = df.sample(n=sample)

    # Mapping numeric targets to their botanical names
    target_mapping = {i: name for i, name in enumerate(iris.target_names)}
    df['target_name'] = df['target'].map(target_mapping)

    # Convert dataframe rows to Pydantic models for strict typing
    samples = [
        IrisSample(**row.to_dict()) 
        for _, row in df.iterrows()
    ]
    
    return samples

if __name__ == "__main__":
    # Quick sanity check
    data = get_iris_data()
    print(f"Loaded {len(data)} samples.")
    print(f"Example prompt input: {data[0].to_natural_language()}")