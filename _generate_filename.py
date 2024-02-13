def generate_filename(timestamp_components):
    timestamp_components = timestamp_components.replace('.mp4', '')
    if "rtsp" in input_name:
        # Extract camera_id from the RTSP URL
        camera_id = timestamp_components.split('/')[-1]

        # Get the current date in DD_MM_YY format
        current_date = datetime.now().strftime("%d_%m_%Y")

        # Get the current time in HH_MM_SS format
        current_time = datetime.now().strftime("%H_%M_%S")

        # Create the filename
        filename = f"{camera_id}_{current_date}_{current_time}_XX.jpg"

        return filename

    elif "footage" in input_name:
        timestamp_components = timestamp_components.split("/")[1].split("_")
        # raw name:  ['1101', '06', '02', '2024', '11', '36', '57.mp4']
        # Extract relevant information
        camera_id = timestamp_components[0]
        date_stamp = f"{timestamp_components[1]}_{timestamp_components[2]}_{timestamp_components[3]}"
        time_start = datetime.strptime(":".join(timestamp_components[4:7]), "%H:%M:%S")

        # Calculate time elapsed since time_start
        current_time = datetime.now()
        time_elapsed = current_time - time_start

        # Extract hours, minutes, and seconds from the timedelta
        hours, remainder = divmod(time_elapsed.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Create the formatted time string
        formatted_time = f"{hours:02d}_{minutes:02d}_{seconds:02d}"

        # Create the filename
        filename = f"{camera_id}_{date_stamp}_{formatted_time}_XX.jpg"
        
        return filename