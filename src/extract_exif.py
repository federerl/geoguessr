import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path


def get_decimal_coordinates(gps_info):
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal format.
    
    Args:
        gps_info: Dictionary containing GPS EXIF data
        
    Returns:
        Tuple of (latitude, longitude) in decimal format, or (None, None)
    """
    def convert_to_degrees(value):
        """Convert GPS coordinates to degrees in float format"""
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)
    
    try:
        lat = gps_info.get('GPSLatitude')
        lat_ref = gps_info.get('GPSLatitudeRef')
        lon = gps_info.get('GPSLongitude')
        lon_ref = gps_info.get('GPSLongitudeRef')
        
        if lat and lat_ref and lon and lon_ref:
            lat_decimal = convert_to_degrees(lat)
            if lat_ref == 'S':
                lat_decimal = -lat_decimal
                
            lon_decimal = convert_to_degrees(lon)
            if lon_ref == 'W':
                lon_decimal = -lon_decimal
                
            return lat_decimal, lon_decimal
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    
    return None, None


def extract_exif_data(image_path):
    """
    Extract GPS coordinates from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing image path, latitude, and longitude if GPS data exists,
        otherwise returns None
    """
    try:
        image = Image.open(image_path)
        exif = image.getexif()
        
        if exif:
            # Extract GPS data
            gps_info = {}
            if exif.get_ifd(0x8825):  # GPS IFD tag
                gps_ifd = exif.get_ifd(0x8825)
                for tag_id, value in gps_ifd.items():
                    tag = GPSTAGS.get(tag_id, tag_id)
                    gps_info[tag] = value
                
                # Convert GPS coordinates to decimal format
                lat, lon = get_decimal_coordinates(gps_info)
                
                # Only return data if we successfully extracted GPS coordinates
                if lat is not None and lon is not None:
                    return {
                        'image': image_path,
                        'latitude': lat,
                        'longitude': lon
                    }
                
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
    
    # Return None if no GPS data found
    return None


def extract_exif_from_directory(directory_path, extensions=None):
    """
    Extract EXIF data from all images in a directory and subdirectories.
    Only includes images with GPS coordinates.
    
    Args:
        directory_path: Path to directory containing images
        extensions: List of file extensions to process (e.g., ['.jpg', '.jpeg', '.png'])
                   If None, processes common image formats
        
    Returns:
        pandas DataFrame containing image path, latitude, and longitude
        for images with GPS data only
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    # Convert extensions to lowercase for case-insensitive matching
    extensions = [ext.lower() for ext in extensions]
    
    exif_records = []
    
    # Walk through directory and all subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in extensions:
                file_path = os.path.join(root, file)
                exif_data = extract_exif_data(file_path)
                # Only add to records if GPS data was found
                if exif_data is not None:
                    exif_records.append(exif_data)
    
    # Create DataFrame
    df = pd.DataFrame(exif_records)
    
    return df


def extract_exif_from_file_list(image_paths):
    """
    Extract EXIF data from a list of image file paths.
    Only includes images with GPS coordinates.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        pandas DataFrame containing image path, latitude, and longitude
        for images with GPS data only
    """
    exif_records = []
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            exif_data = extract_exif_data(image_path)
            # Only add to records if GPS data was found
            if exif_data is not None:
                exif_records.append(exif_data)
        else:
            print(f"Warning: File not found - {image_path}")
    
    # Create DataFrame
    df = pd.DataFrame(exif_records)
    
    return df


# Example usage
if __name__ == "__main__":
    # Example 1: Process all images in a directory
    # df = extract_exif_from_directory('/path/to/your/images')
    
    # Example 2: Process specific image files
    # image_files = [
    #     '/path/to/image1.jpg',
    #     '/path/to/image2.jpg',
    #     '/path/to/image3.png'
    # ]
    # df = extract_exif_from_file_list(image_files)
    
    # Display the results
    # print(df)
    # print(f"\nTotal images with GPS data: {len(df)}")
    
    # Save to CSV if needed
    # df.to_csv('exif_data.csv', index=False)
    
    print("EXIF extraction functions are ready to use!")
    print("\nUsage:")
    print("1. For a directory: df = extract_exif_from_directory('/path/to/images')")
    print("2. For specific files: df = extract_exif_from_file_list([list_of_paths])")
    print("\nReturns DataFrame with columns: image, latitude, longitude")
    print("Note: Only images WITH GPS data are included in the DataFrame")
