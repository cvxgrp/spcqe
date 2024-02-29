import math
from time import perf_counter
from io import StringIO, BytesIO
import requests
import pandas as pd
from typing import Callable, TypedDict, Any, Dict
from functools import wraps
from datetime import datetime
import base64
import zlib


def load_redshift_data(
    siteid: str,
    api_key: str,
    column: str = "ac_power",
    sensor: int | list[int] | None = None,
    tmin: datetime | None = None,
    tmax: datetime | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Queries a SunPower dataset by site id and returns a Pandas DataFrame

    Request an API key by registering at https://pvdb.slacgismo.org and emailing slacgismotutorials@gmail.com with your information and use case.

    Parameters
    ----------
        siteid : str
            site id to query
        api_key : str
            api key for authentication to query data
        column : str
            meas_name to query (default ac_power)
        sensor : int, list[int], optional
            sensor index to query based on number of sensors at the site id (default None)
        tmin : timestamp, optional
            minimum timestamp to query (default None)
        tmax : timestamp, optional
            maximum timestamp to query (default None)
        limit : int, optional
            maximum number of rows to query (default None)
        verbose : bool, optional
            whether to print out timing information (default False)

    Returns
    ------
    df : pd.DataFrame
        Pandas DataFrame containing the queried data
    """

    class QueryParams(TypedDict):
        api_key: str
        siteid: str
        column: str
        sensor: int | list[int] | None
        tmin: datetime | None
        tmax: datetime | None
        limit: int | None

    def decompress_data_to_dataframe(encoded_data):
        # Decode the data
        decoded_data = base64.b64decode(encoded_data)

        # Decompress the data
        decompressed_data = zlib.decompress(decoded_data).decode("utf-8")

        # Attempt to read the decompressed data as CSV
        df = pd.read_csv(StringIO(decompressed_data))

        return df

    def timing(verbose: bool = True) -> Callable:
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = perf_counter()
                result = func(*args, **kwargs)
                end_time = perf_counter()
                execution_time = end_time - start_time
                if verbose:
                    print(f"{func.__name__} took {execution_time:.3f} seconds to run")
                return result

            return wrapper

        return decorator

    @timing(verbose)
    def query_redshift_w_api(
        params: QueryParams, page: int, is_batch: bool = False
    ) -> requests.Response:
        url = "https://api.pvdb.slacgismo.org/v1/query"
        payload = {
            "api_key": params.get("api_key"),
            "siteid": params.get("siteid"),
            "column": params.get("column"),
            "sensor": params.get("sensor"),
            "tmin": str(params.get("tmin")),
            "tmax": str(params.get("tmax")),
            "limit": str(params.get("limit")),
            "page": str(page),
            "is_batch": str(is_batch),
        }

        if sensor is None:
            payload.pop("sensor")
        if tmin is None:
            payload.pop("tmin")
        if tmax is None:
            payload.pop("tmax")
        if limit is None:
            payload.pop("limit")

        response = requests.post(
            url, json=payload, timeout=60 * 5, headers={"Accept-Encoding": "gzip"}
        )

        if response.status_code != 200:
            error = response.json()
            error_msg = error["error"]
            raise Exception(
                f"Query failed with status code {response.status_code}: {error_msg}"
            )
        if verbose:
            print(f"Content size: {len(response.content)}")

        return response

    @timing(verbose)
    def get_query_info(params: QueryParams) -> requests.Response:
        url = "https://api.pvdb.slacgismo.org/v1/query/info/"
        payload = {
            "api_key": params.get("api_key"),
            "siteid": params.get("siteid"),
            "column": params.get("column"),
            "sensor": params.get("sensor"),
            "tmin": str(params.get("tmin")),
            "tmax": str(params.get("tmax")),
            "limit": str(params.get("limit")),
        }

        if sensor is None:
            payload.pop("sensor")
        if tmin is None:
            payload.pop("tmin")
        if tmax is None:
            payload.pop("tmax")
        if limit is None:
            payload.pop("limit")

        response = requests.post(url, json=payload, timeout=60 * 5)

        if response.status_code != 200:
            error = response.json()
            print(error)
            error_msg = error["error"]
            raise Exception(
                f"Query failed with status code {response.status_code}: {error_msg}"
            )

        return response

    def fetch_data(
        query_params: QueryParams, df_list: list[pd.DataFrame], index: int, page: int
    ):
        try:
            response = query_redshift_w_api(query_params, page)
            new_df = decompress_data_to_dataframe(response.content)

            if new_df.empty:
                raise Exception("Empty dataframe returned from query")
            if verbose:
                print(f"Page: {page}, Rows: {len(new_df)}")
            df_list[index] = new_df

        except Exception as e:
            print(e)
            # raise e

    import threading

    data: Dict[str, Any] = {}

    query_params: QueryParams = {
        "api_key": api_key,
        "siteid": siteid,
        "column": column,
        "sensor": sensor,
        "tmin": tmin,
        "tmax": tmax,
        "limit": limit,
    }

    try:
        batch_df: requests.Response = get_query_info(query_params)
        data = batch_df.json()
    except Exception as e:
        raise e
    max_limit = int(data["max_limit"])
    total_count = int(data["total_count"])
    batches = int(data["batches"])
    if verbose:
        print("total number rows for query: ", total_count)
        print("Max number of rows per API call", max_limit)
        print("Total number of batches", batches)

    batch_size = 2  # Max number of threads to run at once (limited by redshift)

    loops = math.ceil(batches / batch_size)

    if batches <= batch_size:
        loops = 1
        batch_size = batches

    running_count = total_count
    page = 0
    df = pd.DataFrame()
    list_of_dfs: list[pd.DataFrame] = []
    for _ in range(loops):
        df_list = [pd.DataFrame() for _ in range(batch_size)]
        page_batch = list(range(page, page + batch_size))
        threads: list[threading.Thread] = []

        # Create threads for each batch of pages
        for i in range(len(page_batch)):
            query_params_copy = query_params.copy()
            if running_count < max_limit:
                query_params_copy["limit"] = running_count
            thread = threading.Thread(
                target=fetch_data, args=(query_params_copy, df_list, i, page_batch[i])
            )
            threads.append(thread)
            thread.start()

            running_count -= max_limit

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Move to the next batch of pages
        page += batch_size

        # Concatenate the dataframes
        valid_df_list = [new_df for new_df in df_list if not new_df.empty]

        list_of_dfs.extend(valid_df_list)

    df = pd.concat(list_of_dfs, ignore_index=True)
    # If any batch returns an empty DataFrame, stop querying
    if df.empty:
        raise Exception("Empty dataframe returned from query")
    return df