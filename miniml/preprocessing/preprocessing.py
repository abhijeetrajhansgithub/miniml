from typing import Dict, List, Any, Optional
import pandas as pd
from miniml.agents.agent_get_refs import AgentGetRefs
from pandas.api.types import is_numeric_dtype

from miniml.inference.engines.engine_frame import GetRefsEngineFrame

class DataPreprocessing:
    def __init__(self,
    data_file_path: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    sheet_name: str | int = 0,
    provider: str | None = None,
    model: str | None = None,
    use_llm: bool = True,
    _parent_base_dir_path: str = None,
    ):
        self.data_file_path = data_file_path
        self.delimiter = delimiter
        self.encoding = encoding
        self.sheet_name = sheet_name

        self.provider = provider
        self.model = model
        self.use_llm = use_llm

        assert _parent_base_dir_path is not None, "Parent base directory path is required"
        self._parent_base_dir_path = _parent_base_dir_path

        self.data = self.load_data()

        self._columns = self.data.columns.tolist()

        # get data info upto .2f
        self._data_info = self.data.describe().round(2)
        print("=" * 50)
        print(self._data_info)
        print("=" * 50)

        self._columnar_inferences: List[GetRefsEngineFrame] = []

        # get desc for a particular column only
        for column in self._columns:
            _column_data = self.data[column].describe().round(2)
            _column_type = self.get_feature_type(column)
            print("=" * 50)
            print(_column_data)
            print(_column_type)
            print("=" * 50)

            if self.use_llm:

                self.get_refs_agent = AgentGetRefs(
                    _parent_base_dir_path=self._parent_base_dir_path,
                    column_data=self.data[column].describe().round(2),
                    column=column,
                    feature_type=_column_type,
                    provider=self.provider,
                    model=self.model
                )

                _ref: Dict[str, Any] = self.get_refs_agent.run()

                if _ref["_instance"] == "GetRefsEngineFrame":
                    if isinstance(_ref["data"], GetRefsEngineFrame):
                        self._columnar_inferences.append(_ref["data"])
        
        print("All columnar inferences completed...")
        for i in self._columnar_inferences:
            print(i)
            print()
                


    def load_data(self) -> pd.DataFrame:
        if self.data_file_path.endswith(".csv"):
            return pd.read_csv(self.data_file_path, delimiter=self.delimiter, encoding=self.encoding)
        elif self.data_file_path.endswith(".xlsx"):
            return pd.read_excel(self.data_file_path, sheet_name=self.sheet_name)
        else:
            raise ValueError("Unsupported file format")
    
    def get_feature_type(self, column: str) -> str:
        series = self.data[column]

        # Encoded categories often have few unique values
        if is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)

            if series.nunique() <= 20 and unique_ratio < 0.05:
                return "categorical"

            return "numeric"

        return "categorical"
        