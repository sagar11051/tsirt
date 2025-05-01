import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import io
from sklearn.cluster import DBSCAN # Import DBSCAN for clustering

# --- Feature-Based Motif Extraction Logic ---

def extract_motif_features(image, nfeatures=1000, eps=30, min_samples=5):
    """
    Extracts a potential motif area based on density of detected features (ORB).

    Args:
        image (numpy.ndarray): The input image (loaded by OpenCV, BGR format).
        nfeatures (int): Maximum number of ORB features to detect.
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other in DBSCAN. (Pixels)
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point in DBSCAN.

    Returns:
        tuple: Bounding box (x, y, w, h) of the likely motif area,
               or None if no suitable area is found.
        int: Total number of keypoints found.
    """
    if image is None:
        return None, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    # ORB (Oriented FAST and Rotated BRIEF) is a good general-purpose feature detector
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Find keypoints (locations of distinctive features) and descriptors (description of the feature)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    total_keypoints = len(keypoints)

    # Need a reasonable number of keypoints to perform clustering meaningfully
    if total_keypoints < 20: # Lowered minimum keypoints slightly
        st.warning(f"Only found {total_keypoints} keypoints. Not enough to identify a motif area.")
        return None, total_keypoints

    # Get keypoint coordinates as a numpy array (DBSCAN works on coordinates)
    keypoint_coords = np.array([kp.pt for kp in keypoints])

    # Use DBSCAN to find clusters of keypoints
    # DBSCAN groups points based on density. Points in sparse areas are marked as noise (-1 label).
    # eps: Determines the maximum distance to consider points neighbors.
    # min_samples: Minimum number of points in a neighborhood for a point to be a core point.
    try:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(keypoint_coords)
        labels = db.labels_

        # Find the most frequent label (excluding noise, which has label -1)
        # Count points in each cluster
        label_counts = {}
        for label in labels:
            if label != -1: # Exclude noise points
                label_counts[label] = label_counts.get(label, 0) + 1

        if not label_counts:
            st.warning("DBSCAN found no significant clusters of keypoints (all points considered noise or too sparse).")
            return None, total_keypoints

        # Find the label of the largest cluster by count
        largest_cluster_label = max(label_counts, key=label_counts.get)

        # Get the keypoints belonging to the largest cluster
        largest_cluster_points = keypoint_coords[labels == largest_cluster_label]

        # Need a minimum number of points *in the cluster* to consider it a valid motif area
        if largest_cluster_points.shape[0] < min_samples: # Check if the cluster size meets min_samples criterion
             st.warning(f"Largest cluster size ({largest_cluster_points.shape[0]}) is less than the minimum samples required ({min_samples}).")
             return None, total_keypoints


        # Calculate the bounding box around the keypoints in the largest cluster
        x_min = int(np.min(largest_cluster_points[:, 0]))
        y_min = int(np.min(largest_cluster_points[:, 1]))
        x_max = int(np.max(largest_cluster_points[:, 0]))
        y_max = int(np.max(largest_cluster_points[:, 1]))

        # Sanity checks on the calculated bounding box size
        box_width = x_max - x_min
        box_height = y_max - y_min
        img_area = image.shape[0] * image.shape[1]
        box_area = box_width * box_height

        min_box_dimension = 20 # Minimum pixel dimension for the box sides
        max_box_area_ratio = 0.6 # Max area of the box relative to the image area

        if box_width < min_box_dimension or box_height < min_box_dimension or \
           box_area < 100 or box_area > img_area * max_box_area_ratio:
             st.warning(f"Calculated bounding box from cluster is an unreasonable size ({box_width}x{box_height}, Area: {box_area}). Skipping extraction.")
             return None, total_keypoints


        motif_box = (x_min, y_min, box_width, box_height)

        return motif_box, total_keypoints

    except Exception as e:
        st.error(f"An error occurred during feature detection or clustering: {e}")
        # Provide a more informative error if possible, but general exception catch is okay for now
        return None, total_keypoints


# --- Extraction Helper Function (Reused) ---

def extract_centered_square_tile(image, motif_box, padding_percent=10):
    """
    Extracts a centered square tile containing the motif_box.
    (This function remains largely the same as it works on a given box)

    Args:
        image (numpy.ndarray): The original color image (BGR format).
        motif_box (tuple): Bounding box (x, y, w, h) of the motif.
        padding_percent (int): Percentage of tile size to add as padding.

    Returns:
        numpy.ndarray: The extracted square tile (BGR format), or None if extraction fails.
        tuple: The calculated crop coordinates (x_min, y_min, x_max, y_max) before boundary check.
        tuple: The final crop coordinates (x_min, y_min, x_max, y_max) after boundary check.
    """
    img_height, img_width = image.shape[:2]
    x, y, w, h = motif_box

    center_x = x + w / 2
    center_y = y + h / 2

    # Determine the size of the square based on the larger dimension of the motif box
    tile_size = max(w, h)

    # Add padding to the tile size
    padding = tile_size * (padding_percent / 100.0)
    tile_size_padded = int(tile_size + 2 * padding)

    # Calculate potential crop coordinates based on the center and padded tile size
    crop_x_min = int(center_x - tile_size_padded / 2)
    crop_y_min = int(center_y - tile_size_padded / 2)
    crop_x_max = int(center_x + tile_size_padded / 2)
    crop_y_max = int(center_y + tile_size_padded / 2)

    # Store calculated crop box info to return
    calculated_box = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)

    # Ensure the crop coordinates are within the image boundaries
    final_crop_x_min = max(0, crop_x_min)
    final_crop_y_min = max(0, crop_y_min)
    final_crop_x_max = min(img_width, crop_x_max)
    final_crop_y_max = min(img_height, crop_y_max)

    final_box = (final_crop_x_min, final_crop_y_min, final_crop_x_max, final_crop_y_max)

    # Check if the final crop area is valid (width and height > 0)
    if final_crop_x_min >= final_crop_x_max or final_crop_y_min >= final_crop_y_max:
        # Return None for tile and the boxes if the crop is invalid
        return None, calculated_box, final_box

    # Crop the image using numpy slicing [ymin:ymax, xmin:xmax]
    # Ensure the slicing creates a valid tile
    try:
        tile = image[final_crop_y_min:final_crop_y_max, final_crop_x_min:final_crop_x_max]
        # Check if the resulting tile is empty (e.g., if crop dimensions were somehow zero after bounds check)
        if tile is None or tile.size == 0:
             return None, calculated_box, final_box
    except Exception as e:
        st.error(f"Error during image slicing: {e}")
        return None, calculated_box, final_box


    return tile, calculated_box, final_box


# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("👕 T-Shirt Motif Extractor (Feature-Based)")

st.write("Upload a t-shirt image. This app attempts to find the motif by detecting features and clustering them based on spatial density.")

# Sidebar for controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Choose a t-shirt image...", type=["jpg", "jpeg", "png"])
padding_percent = st.sidebar.slider("Padding around motif (%)", 0, 50, 10)

st.sidebar.write("---") # Separator for parameters
st.sidebar.subheader("Feature Detection Parameters")
# Control the number of ORB features to detect. More features might find smaller details.
orb_features = st.sidebar.slider("ORB Max Features", 100, 5000, 1500, help="Maximum number of features to detect. Adjust based on image complexity.")

st.sidebar.subheader("Clustering Parameters (DBSCAN)")
# DBSCAN Epsilon: Max distance between points for them to be considered neighbors. Tune based on image resolution.
dbscan_eps = st.sidebar.slider("DBSCAN Epsilon (pixels)", 5, 100, 30, help="Maximum distance for points to be considered neighbors. Increase for lower resolution/larger features, decrease for higher resolution/smaller features.")
# DBSCAN Min Samples: Minimum number of points to form a dense region (a cluster).
dbscan_min_samples = st.sidebar.slider("DBSCAN Min Samples per Cluster", 2, 30, 7, help="Minimum number of features required to form a cluster. Increase for noisier images.")


# Main area layout
col1, col2 = st.columns(2)

# Placeholders for dynamic content
original_image_placeholder = col1.empty()
processing_status_placeholder = col2.empty()
extracted_tile_placeholder = col2.empty()
download_button_placeholder = col2.empty()
info_messages_placeholder = col2.empty() # Placeholder for messages

if uploaded_file is not None:
    # Read the uploaded image file bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode the image using OpenCV
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Check if image decoding was successful
    if opencv_image is None:
        col1.error("Error: Could not decode the uploaded image.")
    else:
        # Convert color from BGR (OpenCV default) to RGB (Streamlit/PIL default) for display
        display_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        col1.subheader("Original Image")
        original_image_placeholder.image(display_image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        st.sidebar.write("---") # Separator
        if st.sidebar.button("✨ Extract Motif"):
            # Clear previous results and messages
            processing_status_placeholder.empty()
            extracted_tile_placeholder.empty()
            download_button_placeholder.empty()
            info_messages_placeholder.empty()

            with col2: # Use 'with' notation for cleaner status updates in the column
                processing_status_placeholder.subheader("Processing...")
                progress_bar = st.progress(0) # Add progress bar

                # 1. Find Motif Area using Features and Clustering
                progress_bar.progress(25, text=f"Detecting up to {orb_features} features...")
                # Pass the OpenCV image (BGR) and user parameters to the feature extraction function
                motif_box, total_keypoints = extract_motif_features(
                    opencv_image,
                    nfeatures=orb_features,
                    eps=dbscan_eps,
                    min_samples=dbscan_min_samples
                )
                info_messages_placeholder.write(f"Found {total_keypoints} keypoints in total.")

                progress_bar.progress(60, text="Clustering features and selecting motif area...")

                if motif_box:
                    info_messages_placeholder.write(f"Selected motif area bounding box: {motif_box}")
                    # 2. Extract Tile using the bounding box
                    progress_bar.progress(75, text="Extracting tile...")
                    # Pass the original BGR image and the motif_box to the extraction function
                    square_tile_bgr, calc_box, final_box = extract_centered_square_tile(opencv_image, motif_box, padding_percent)

                    info_messages_placeholder.write(f"Calculated square crop box (before boundary check): [{calc_box[0]}, {calc_box[1]}, {calc_box[2]}, {calc_box[3]}]")
                    info_messages_placeholder.write(f"Final crop box (after boundary check): [{final_box[0]}, {final_box[1]}, {final_box[2]}, {final_box[3]}]")
                    progress_bar.progress(85, text="Preparing display...")


                    if square_tile_bgr is not None and square_tile_bgr.size > 0:
                        extracted_tile_placeholder.subheader("Extracted Motif Tile")
                        # Convert tile to RGB for display in Streamlit
                        square_tile_rgb = cv2.cvtColor(square_tile_bgr, cv2.COLOR_BGR2RGB)
                        extracted_tile_placeholder.image(square_tile_rgb, caption="Extracted Motif", use_container_width=True)

                        # Provide download button
                        # Convert tile back to format suitable for download (e.g., PNG)
                        pil_img = Image.fromarray(square_tile_rgb) # Convert numpy array (RGB) to PIL Image
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        byte_im = buf.getvalue()

                        # Get original filename without extension
                        file_name_base = uploaded_file.name.rsplit('.', 1)[0]

                        download_button_placeholder.download_button(
                            label="⬇️ Download Tile (PNG)",
                            data=byte_im,
                            file_name=f"{file_name_base}_tile.png",
                            mime="image/png"
                        )
                        progress_bar.progress(100, text="Done!")
                        processing_status_placeholder.empty() # Clear "Processing..."
                        st.success("Extraction complete!")

                    else:
                         processing_status_placeholder.empty() # Clear "Processing..."
                         st.error("Could not extract tile (result was empty or invalid after cropping).")
                         progress_bar.empty() # Remove progress bar on error

                else:
                    # If motif_box is None, detection failed.
                    # The extract_motif_features function already prints a warning if applicable.
                    processing_status_placeholder.empty() # Clear "Processing..."
                    progress_bar.empty() # Remove progress bar
                    extracted_tile_placeholder.empty() # Clear any previous tile
                    download_button_placeholder.empty() # Clear download button


else:
    col1.info("Upload an image using the sidebar to begin.")
    col2.info("Extracted motif will appear here.")