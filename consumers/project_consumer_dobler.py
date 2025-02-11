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

# Import external packages
from dotenv import load_dotenv
import matplotlib.pyplot as plt

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
    ax.plot(timestamps, sentiments, marker='o', linestyle='-', color='b')
    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Analysis Over Time")
    plt.xticks(rotation=45)
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

        if isinstance(message_dict, dict):
            timestamp = message_dict.get("timestamp", "unknown")
            sentiment = float(message_dict.get("sentiment", 0))

            timestamps.append(timestamp)
            sentiments.append(sentiment)

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
        logger.info(f"Kafka consumer for topic '{topic}' closed.")

    logger.info(f"END consumer for topic '{topic}' and group '{group_id}'")

if __name__ == "__main__":
    main()
    plt.ioff()  # Turn off interactive mode after completion
    plt.show()
