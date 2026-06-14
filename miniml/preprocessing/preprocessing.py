from typing import Dict, List, Any, Literal, Optional
import pandas as pd
import os

from miniml.agents.agent_get_refs import AgentGetRefs
from miniml.agents.agent_find_target import AgentFindTarget
from miniml.agents.agent_apply_refs import AgentApplyRefs

from pandas.api.types import is_numeric_dtype
from miniml.tools.toolreg import Tool

from miniml.inference.engines.engine_frame import GetRefsEngineFrame, FindTargetEngineFrame

class DataPreprocessing:
    def __init__(self,
    data_file_path: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    sheet_name: str | int = 0,
    provider: str | None = None,
    model: str | None = None,
    use_llm: bool = True,
    target_column: str | None = None,
    output_path: str | None = None,
    tools: List[Tool] = None,
    _parent_base_dir_path: str | None = None,
    ):
        self.data_file_path = data_file_path
        self.delimiter = delimiter
        self.encoding = encoding  # file encoding
        self.sheet_name = sheet_name

        self.provider = provider
        self.model = model
        self.use_llm = use_llm
        self.target_column = target_column
        self.output_path = output_path
        self.tools = tools

        assert _parent_base_dir_path is not None, "Parent base directory path is required"
        self._parent_base_dir_path = _parent_base_dir_path

        self.data: pd.DataFrame = self.load_data()

        self._columns: List[str] = self.data.columns.tolist()

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
                    model=self.model,
                    tools=self.tools
                )

                _ref: Dict[str, Any] = self.get_refs_agent.run()

                if _ref["_instance"] == "GetRefsEngineFrame":
                    if isinstance(_ref["data"], GetRefsEngineFrame):
                        self._columnar_inferences.append(_ref["data"])
        
        print("All columnar inferences completed...")
        for i in self._columnar_inferences:
            print(i)
            print()

        self._list_of_transformations: List[str] = []

        for inf in self._columnar_inferences:
            _impuration_strategy = inf.imputation_strategy
            _outlier_strategy = inf.outlier_strategy

            if _impuration_strategy not in self._list_of_transformations:
                self._list_of_transformations.append(str(_impuration_strategy))
            
            if _outlier_strategy not in self._list_of_transformations:
                self._list_of_transformations.append(str(_outlier_strategy))
        

        # Agent Target Finder
        if self.target_column is None:
            print("Target name is not provided...")
            # TODO: Implement target finder agent
            self.find_target_agent = AgentFindTarget(
                column_inferences=self._columnar_inferences,
                provider=self.provider,
                model=self.model,
                tools=self.tools,
                _parent_base_dir_path=self._parent_base_dir_path
            )

            _target: Dict[str, Any] = self.find_target_agent.run()

            if _target["_instance"] == "FindTargetEngineFrame":
                if isinstance(_target["data"], FindTargetEngineFrame):
                    self.target_column = _target["data"].target_column
                    print(f"Target column is: {self.target_column}")
        else:
            print("Target name is provided...")

        
        for inference in self._columnar_inferences:
            self.apply_refs_agent = AgentApplyRefs(
                column_inference=inference,
                main_dataframe=self.data,
                provider=self.provider,
                model=self.model,
                tools=self.tools,
                _parent_base_dir_path=self._parent_base_dir_path
            )
            self.data: pd.DataFrame = self.apply_refs_agent.run()
        

        print("Data processed successfully!")

        if not self.output_path:
            self.data.to_csv("data-processed.csv", index=False)

        else:
            path_type = self._resolve_path(self.output_path)

            if path_type == "abs":
                output_file = os.path.join(
                    self.output_path,
                    "data-processed.csv"
                )

            else:
                # save to Downloads
                output_file = (
                    Path.home()
                    / "Downloads"
                    / "data-processed.csv"
                )

            self.data.to_csv(output_file, index=False)
    
    def get_modified_data(self) -> pd.DataFrame:
        return self.data if isinstance(self.data, pd.DataFrame) else None

        
    def _resolve_path(self, path: str) -> Literal["abs", "rel"]:
        return "abs" if os.path.isabs(path) else "rel"

    
    def encode_data(self) -> None:
        pass

                
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

    def get_column_types(self) -> Dict[str, List[str]]:
        numeric = []
        categor = []

        for column in self._columns:
            _column_type = self.get_feature_type(column)

            if _column_type == "numeric":
                numeric.append(column)
            elif _column_type == "categorical":
                categor.append(column)
        
        return {
            "numeric": numeric,
            "categorical": categor
        }


    def get_transformations_applied(self) -> List[str]:
        return self._list_of_transformations or []
        