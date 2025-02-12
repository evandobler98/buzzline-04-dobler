#####################################
# project_consumer_dobler.py
#
# Consume JSON messages from a Kafka topic
# and visualize sentiment trends over time.
#####################################

# Import Python Standard Library modules
import os
import json
from collections import deque  # Use deque for efficient list operations
from datetime import datetime  # Convert timestamps to readable format

# Import external packages
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np  # For smoothing data

# Import local utility functions
from utils.utils_consumer import create_kafka_consumer
from utils.utils_logger import logger

#####################################
# Load Environment Variables
#####################################

load_dotenv()

def get_kafka_topic() -> str:
    """Fetch Kafka topic from environment or use default."""
    topic = os.getenv("BUZZ_TOPIC", "unknown_topic")
    logger.info(f"Kafka topic: {topic}")
    return topic

def get_kafka_consumer_group_id() -> str:
    """Fetch Kafka consumer group id from environment or use default."""
    group_id = os.getenv("BUZZ_CONSUMER_GROUP_ID", "default_group")
    logger.info(f"Kafka consumer group id: {group_id}")
    return group_id

#####################################
# Set up Data Structures for Sentiment Tracking
#####################################

MAX_POINTS = 50  # Limit points for better visualization
timestamps = deque(maxlen=MAX_POINTS)
sentiments = deque(maxlen=MAX_POINTS)

#####################################
# Set up Live Visualization
#####################################

fig, ax = plt.subplots()
plt.ion()  # Turn on interactive mode

def update_chart():
    """Update the live chart with the latest sentiment trends."""
    ax.clear()

    # Convert deque to lists for plotting
    time_labels = list(timestamps)
    sentiment_values = list(sentiments)

    # Plot raw sentiment values
    ax.plot(time_labels, sentiment_values, marker='o', linestyle='-', color='b', label="Sentiment Score")

    # Apply moving average smoothing if we have enough points
    if len(sentiment_values) > 5:
        smooth_sentiments = np.convolve(sentiment_values, np.ones(5)/5, mode='valid')
        ax.plot(time_labels[-len(smooth_sentiments):], smooth_sentiments, linestyle='-', color='r', label="Smoothed")

    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Analysis Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

def process_message(message: str) -> None:
    """
    Process a single JSON message from Kafka and update the sentiment trend.
    """
    try:
        message_dict = json.loads(message)
        logger.info(f"Processed JSON message: {message_dict}")

        # Validate message contains required fields
        if not isinstance(message_dict, dict):
            logger.error(f"Invalid message format: {message_dict}")
            return

        timestamp = message_dict.get("timestamp")
        sentiment = message_dict.get("sentiment")

        if timestamp is None or sentiment is None:
            logger.warning(f"Missing data fields in message: {message_dict}")
            return

        # Convert timestamp to readable format if it's in Unix time
        try:
            timestamp = datetime.fromtimestamp(int(timestamp)).strftime("%H:%M:%S")
        except ValueError:
            logger.warning(f"Invalid timestamp value: {timestamp}")
            return

        # Convert sentiment to float safely
        try:
            sentiment = float(sentiment)
        except ValueError:
            logger.warning(f"Invalid sentiment value: {sentiment}")
            return

        # Append data for visualization
        timestamps.append(timestamp)
        sentiments.append(sentiment)

        # Update the live chart
        update_chart()

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON message: {message}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")

def main() -> None:
    """Main function to consume Kafka messages and update visualization."""
    logger.info("START consumer.")
    
    topic = get_kafka_topic()
    group_id = get_kafka_consumer_group_id()
    logger.info(f"Consumer: Topic '{topic}' and group '{group_id}'...")

    consumer = create_kafka_consumer(topic, group_id)
    
    logger.info(f"Polling messages from topic '{topic}'...")
    try:
        for message in consumer:
            message_str = message.value
            logger.debug(f"Received message at offset {message.offset}: {message_str}")
            process_message(message_str)
    except KeyboardInterrupt:
        logger.warning("Consumer interrupted by user.")
    except Exception as e:
        logger.error(f"Error while consuming messages: {e}")
    finally:
        consumer.close()
        plt.close(fig)  # Ensure figure closes properly
        logger.info(f"Kafka consumer for topic '{topic}' closed.")

    logger.info(f"END consumer for topic '{topic}' and group '{group_id}'")

if __name__ == "__main__":
    main()
    plt.ioff()  # Turn off interactive mode after completion
    plt.show()
