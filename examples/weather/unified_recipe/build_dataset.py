import argparse
import os
import pandas as pd
import zarr
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.metrics.metric import Metrics
from loguru import logger

from earth2studio import data as earth2studio_data
from earth2studio.utils import type as earth2studio_type
from transform.transform import transform_registry


def build_dataset(
    source: earth2studio_data.DataSource,
    time: earth2studio_type.TimeArray,
    variable: earth2studio_type.VariableArray,
    dataset: zarr.Array,
    transform: None = None,
    apache_beam_options: None = None,
):
    """
    Utility function to fetch data for models and load data on the target device.
    """

    # Set default options
    if apache_beam_options is None:
        logger.info(
            "No Apache Beam options provided, using default options (Single-threaded,local runner)"
        )
        apache_beam_options = PipelineOptions()

    # Check dataset has correct shape
    if dataset.shape[0] != len(time) or dataset.shape[1] != len(variable):
        raise ValueError(
            f"Dataset shape {dataset.shape} does not match time {len(time)} and "
            f"variable {len(variable)} shape"
        )

    # Fetch data on first time step to make sure all datasets in cache are created
    for var in variable:
        source.fetch_array(time[0], var)

    # Create elements for Apache Beam pipeline
    elements = [(i, t, variable) for i, t in enumerate(time)]

    # Make fetch data DoFn
    class FetchDataDoFn(beam.DoFn):
        def __init__(self, source, dataset, total, report_every=1000, transform=None):
            self.source = source
            self.dataset = dataset
            self.total = total
            self.report_every = report_every
            self.transform = transform
            self.progress = Metrics.counter(self.__class__, 'progress')

        def process(self, element):
            t_index, time, variable = element
            for i, var in enumerate(variable):
                data = self.source.fetch_array(time, var)
                if self.transform is not None:
                    data = self.transform(time, var, data)
                self.dataset[t_index, i, :, :] = data
            self.progress.inc()
            yield (time, variable)

    # Run the pipeline
    logger.info("Starting Apache Beam pipeline")
    with beam.Pipeline(options=apache_beam_options) as p:
        arco_chunks = (
            p
            | "Create elements" >> beam.Create(elements)
            | "Fetch data" >> beam.ParDo(
                FetchDataDoFn(source, dataset, len(time), report_every=10, transform=transform)
            )
        )


def main():
    parser = argparse.ArgumentParser(description='Build a dataset from data sources.')
    
    # Custom arguments for your data processing
    parser.add_argument(
        '--data_source', type=str, default='ARCO', help='Data source to use. Default is ARCO.'
    )
    parser.add_argument(
        '--start_time', type=str, default='2000-01-01',
        help='Start time in the format YYYY-MM-DD. Default is 2000-01-01.'
    )
    parser.add_argument(
        '--frequency', type=str, default='6h', help='Frequency of the data. Default is 6h.'
    )
    parser.add_argument(
        '--periods', type=int, default=1000,
        help='Number of periods for the time range. Default is 1000.'
    )
    parser.add_argument(
        '--variables', type=str, nargs='+', default=['u10m', 'v10m'],
        help='List of variables to fetch. Default is [u10m, v10m].'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', default=['downsample'],
        help='List of transforms to apply. Default is [downsample].'
    )
    parser.add_argument(
        '--dimensions', type=int, nargs='+', default=[1000, 2, 721, 1440],
        help='Dimensions of the Zarr array. Default is [1000, 2, 721, 1440].'
    )
    parser.add_argument(
        '--subgroup_name', type=str, default='predicted_variables',
        help='Subgroup name for the dataset. Default is predicted_variables.'
    )
    parser.add_argument(
        '--zarr_cache', type=str, default='my_zarr_cache.zarr',
        help='Name of the Zarr cache. Default is my_zarr_cache.zarr.'
    )
    parser.add_argument(
        '--zarr_dataset', type=str, default='my_dataset.zarr',
        help='Name of the Zarr dataset. Default is my_dataset.zarr.'
    )

    # Apache Beam options
    parser.add_argument(
        '--beam_options', nargs=argparse.REMAINDER, default=[
            '--runner=DirectRunner',
            '--direct_num_workers=8',
            '--direct_running_mode=multi_processing',
        ], help='Apache Beam options. Default is DirectRunner with 8 workers and multi-processing.'
    )

    args = parser.parse_args()

    # Create zarr cache
    zarr_cache = zarr.open(args.zarr_cache, mode="a")

    # Create the data source
    data_source = getattr(earth2studio_data, args.data_source)(zarr_cache=zarr_cache)
    logger.info("Fetched data from data source: {}".format(args.data_source))

    # Apache Beam options
    options = PipelineOptions(args.beam_options)

    # Make Store Dataset
    time = pd.date_range(args.start_time, freq=args.frequency, periods=args.periods)

    # Check if the file exists
    if os.path.exists(args.zarr_dataset):
        zarr_dataset = zarr.open(args.zarr_dataset, mode="a")
    else:
        zarr_dataset = zarr.open(args.zarr_dataset, mode="w")  # create a new Zarr store
    logger.info("Created Zarr dataset: {}".format(args.zarr_dataset))

    for transform_name in args.transforms:
        
        # TODO (@saikrishnanc-nv): Implement applying transforms
        # selected_transform = transform_registry[transform_name]

        dataset = zarr_dataset.create_dataset(
            args.subgroup_name,
            shape=args.dimensions,
            chunks=(1, len(args.variables), 721, 1440),
            dtype="f4"
        )
        build_dataset(
            data_source,
            time,
            args.variables,
            dataset,
            apache_beam_options=options,
            transform=None
        )

    logger.info("Built dataset!")

if __name__ == "__main__":
    main()
