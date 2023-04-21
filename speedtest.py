import tensorflow as tf
import datetime
import time

def time_configs(dataset, batch_sizes, batch_numbers, repetitions, filename):
    """Function to run time measurement on combinations of batch_sizes and batch_numbers for each repetition
    and store the results in a file.

    Args:
    - dataset: tf.data.Dataset, the dataset to use for testing
    - batch_sizes: list of integers, the different batch sizes to test
    - batch_numbers: list of integers, the different numbers of batches to test
    - repetitions: list of integers, the number of times to repeat the test with the same dataset
    - filename: str, the name of the file to store the results
    """

    # Open the file for writing
    f = open(filename, 'w')
    
    # Iterate over the different batch sizes and numbers
    for bs in batch_sizes:
        for bn in batch_numbers:
            # Calculate the number of images in the batches
            n_images = bs * bn

            # Iterate over the repetitions
            for r in repetitions:
                # Take a certain number of batches from the dataset
                dset1 = dataset
                dset2 = dset1.batch(bs)
                dset3 = dset2.take(bn)
                
                # Read the images and time it
                start_time = time.time()
                for batch in dset3:
                    null_file=open("/dev/null", mode='w')
                    for image in batch:
                        tf.print(tf.shape(image), output_stream=null_file)
                end_time = time.time()

                # Calculate the elapsed time and the speed in images per second
                elapsed_time = end_time - start_time
                speed = n_images / elapsed_time
                
                # Write the results to the file
                f.write(f"Batch size: {bs}, Batch number: {bn}, Repetition: {r}\n")
                f.write(f"Elapsed time (s): {elapsed_time:.2f}, Speed (images/s): {speed:.2f}\n\n")
    
    # Close the file
    f.close()
