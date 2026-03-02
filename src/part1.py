import os
import time
import shutil
import pandas as pd


#Preview the original dataset
def preview_csv(csv_path, n=5):
    print("1) PREVIEW CSV (head + info)")
    print("==============================")

    df = pd.read_csv(csv_path)
    print("\nFirst 5 rows:")
    print(df.head(n))

    print("\nDataFrame info():")
    df.info()

    return df 

#X10 AND X100 DATA SIZE (create temporary copies in a temp folder)

def scale_up_data(x1_src, temp_folder, scale, datatype):
  
    # clean old folder if exists
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    start = time.perf_counter()
    for i in range(scale):
        dst_file = os.path.join(temp_folder, f"data_{i}.{datatype}")
        shutil.copy2(x1_src, dst_file)
    end = time.perf_counter()

    return end - start


# MEASURES FOR BENCHMARKING

# File's reading time for csv
def measure_read_time_csv(file_list):
    start = time.perf_counter()
    for f in file_list:
        df = pd.read_csv(f)
        del df
    end = time.perf_counter()
    return end - start

# File's reading time for parquet
def measure_read_time_parquet(file_list):
    start = time.perf_counter()
    for f in file_list:
        df = pd.read_parquet(f, engine="pyarrow")
        del df
    end = time.perf_counter()
    return end - start

# File's size
def measure_data_size_bytes(path):
    if os.path.isfile(path):
        return os.path.getsize(path)

    total = 0
    for root, dirs, files in os.walk(path): 
        for name in files:
            filePath = os.path.join(root, name)
            total += os.path.getsize(filePath)
    return total

# Scaling time
def measure_writing_time_for_scale_up(src_file, temp_folder, scale, dataType):
    return scale_up_data(src_file, temp_folder, scale, dataType)

# CONVERT CSV TO PARQUET

def convert_csv_to_parquet_and_measure_time(csv_path, parquet_path, compression="snappy"):

    start = time.perf_counter()

    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression=compression)

    del df
    end = time.perf_counter()

    return end - start


# APPLICATION

def main():

    csv_path = "data/all_stocks_5yr.csv"      
    benchmarking_output = "outputs/part1_benchmarking.csv"

    # temp folders 
    temp_root = "temp_folder"
    csv_10x_folder = os.path.join(temp_root, "csv_10x")
    csv_100x_folder = os.path.join(temp_root, "csv_100x")
    parquet_10x_folder = os.path.join(temp_root, "parquet_10x")
    parquet_100x_folder = os.path.join(temp_root, "parquet_100x")

    parquet_1x_file = os.path.join(temp_root, "data_1x_snappy.parquet")
    compression = "snappy"

    # create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # check csv exists
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        return

    # read csv file, display first 5 rows, show df.info()
    preview_csv(csv_path)

    results = [] #store benchmarking results

    # measure reading time and data size of csv files for x1, x10, x100
    print("\n==============================")
    print("A. CSV: Measure size + read time (1x, 10x, 100x)")
    print("==============================")

    # --- CSV 1x ---
    csv_1x_size = measure_data_size_bytes(csv_path)
    csv_1x_read_time = measure_read_time_csv([csv_path])

    results.append({
        "format": "csv",
        "scale": "1x",
        "compression": "none",
        "path": csv_path,
        "write_time_sec": 0.0,  # because 1x csv already exists
        "size_bytes": csv_1x_size,
        "read_time_sec": csv_1x_read_time
    })

    # --- CSV 10x ---
    csv_10x_write = measure_writing_time_for_scale_up(csv_path, csv_10x_folder, 10, "csv")
    csv_10x_size = measure_data_size_bytes(csv_10x_folder)
    csv_10x_files = [os.path.join(csv_10x_folder, f) for f in os.listdir(csv_10x_folder)]
    csv_10x_read = measure_read_time_csv(csv_10x_files)

    results.append({
        "format": "csv",
        "scale": "10x",
        "compression": "none",
        "path": csv_10x_folder,
        "write_time_sec": csv_10x_write,
        "size_bytes": csv_10x_size,
        "read_time_sec": csv_10x_read
    })

    # --- CSV 100x ---
    csv_100x_write = measure_writing_time_for_scale_up(csv_path, csv_100x_folder, 100, "csv")
    csv_100x_size = measure_data_size_bytes(csv_100x_folder)
    csv_100x_files = [os.path.join(csv_100x_folder, f) for f in os.listdir(csv_100x_folder)]
    csv_100x_read = measure_read_time_csv(csv_100x_files)

    results.append({
        "format": "csv",
        "scale": "100x",
        "compression": "none",
        "path": csv_100x_folder,
        "write_time_sec": csv_100x_write,
        "size_bytes": csv_100x_size,
        "read_time_sec": csv_100x_read
    })

    # Convert csv to parquet and measure conversion time
   
    print("\n==============================")
    print("Convert CSV -> Parquet (1x) and measure conversion time")
    print("==============================")

    os.makedirs(temp_root, exist_ok=True)
    parquet_1x_write = convert_csv_to_parquet_and_measure_time(
        csv_path, parquet_1x_file, compression=compression
    )

    # Measure reading time and data size of parquet files for x1, x10, x100
   
    print("\n==============================")
    print("B. Parquet: Measure size + read time (1x, 10x, 100x)")
    print("================================")

    # --- Parquet 1x ---
    parquet_1x_size = measure_data_size_bytes(parquet_1x_file)
    parquet_1x_read = measure_read_time_parquet([parquet_1x_file])

    results.append({
        "format": "parquet",
        "scale": "1x",
        "compression": compression,
        "path": parquet_1x_file,
        "write_time_sec": parquet_1x_write,  # conversion time
        "size_bytes": parquet_1x_size,
        "read_time_sec": parquet_1x_read
    })

    # x10 and x100 the parquet file (temporary copies)
    
    # --- Parquet 10x ---
    parquet_10x_write = measure_writing_time_for_scale_up(parquet_1x_file, parquet_10x_folder, 10, "parquet")
    parquet_10x_size = measure_data_size_bytes(parquet_10x_folder)
    parquet_10x_files = [os.path.join(parquet_10x_folder, f) for f in os.listdir(parquet_10x_folder)]
    parquet_10x_read = measure_read_time_parquet(parquet_10x_files)

    results.append({
        "format": "parquet",
        "scale": "10x",
        "compression": compression,
        "path": parquet_10x_folder,
        "write_time_sec": parquet_10x_write,
        "size_bytes": parquet_10x_size,
        "read_time_sec": parquet_10x_read
    })

    # --- Parquet 100x ---
    parquet_100x_write = measure_writing_time_for_scale_up(parquet_1x_file, parquet_100x_folder, 100, "parquet")
    parquet_100x_size = measure_data_size_bytes(parquet_100x_folder)
    parquet_100x_files = [os.path.join(parquet_100x_folder, f) for f in os.listdir(parquet_100x_folder)]
    parquet_100x_read = measure_read_time_parquet(parquet_100x_files)

    results.append({
        "format": "parquet",
        "scale": "100x",
        "compression": compression,
        "path": parquet_100x_folder,
        "write_time_sec": parquet_100x_write,
        "size_bytes": parquet_100x_size,
        "read_time_sec": parquet_100x_read
    })

    # Save results

    df_results = pd.DataFrame(results)
    df_results.to_csv(benchmarking_output, index=False)

    print("\n==============================")
    print("DONE! Results saved")
    print("==============================")
    print(f"Results file: {benchmarking_output}")
    print("\nPreview results:")
    print(df_results)

    # Delete temp data
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
        print("\nx10/x100 files deleted.")


if __name__ == "__main__":
    main()