import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableAnalyzer:
    def __init__(self):
        logger.info("Initializing TAPAS model...")
        self.model_name = "google/tapas-large-finetuned-wtq"
        self.tokenizer = TapasTokenizer.from_pretrained(self.model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(self.model_name)
        logger.info("TAPAS model ready!")

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV file and return as pandas DataFrame"""
        try:
            df = pd.read_csv(filepath)
            # Remove any unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            # Convert all numeric values to formatted strings
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].apply(lambda x: f"{x:,}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def query_table(self, table: pd.DataFrame, query: str) -> str:
        """Query the table using TAPAS model"""
        try:
            # Convert DataFrame to string types
            table = table.astype(str)
            
            # Clean up the table for better querying
            table = table.replace('nan', '')
            
            # Encode the question and table
            inputs = self.tokenizer(
                table=table,
                queries=[query],
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Disable gradient computation
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Detach tensors before prediction
                logits = outputs.logits.detach()
                logits_agg = outputs.logits_aggregation.detach()
                
                predicted_answer_coords = self.tokenizer.convert_logits_to_predictions(
                    inputs,
                    logits,
                    logits_agg
                )
            
            # Extract coordinates and aggregation indices
            coords, agg = predicted_answer_coords
            
            # Get answer from table
            if coords and coords[0]:
                answer = self._get_answer_from_coords(table, coords[0], agg[0])
                return f"{str(answer).strip()}"
            else:
                return "Could not find an answer in the table"

        except Exception as e:
            logger.error(f"Error querying table: {e}")
            return f"Error: {str(e)}"

    def _get_answer_from_coords(self, table: pd.DataFrame, coords: list, agg_index: int) -> str:
        """Extract answer from table using coordinates and aggregation"""
        try:
            if not coords:
                return "No answer found"

            # Get values from coordinates
            values = []
            for coord in coords:
                # Use iloc for integer-based indexing
                value = table.iloc[coord[0], coord[1]]
                # Try to convert string numbers back to float for aggregation
                try:
                    # Remove commas and convert to float
                    value = float(str(value).replace(',', ''))
                except (ValueError, AttributeError):
                    pass
                values.append(value)

            # Apply aggregation if needed
            if agg_index == 0:  # NONE
                if len(values) == 1:
                    if isinstance(values[0], (int, float)):
                        return f"{values[0]:,.0f}"
                    return str(values[0])
                return ", ".join([str(v) for v in values])
            elif agg_index == 1:  # SUM
                return f"{sum(values):,.0f}"
            elif agg_index == 2:  # AVERAGE
                return f"{sum(values) / len(values):,.2f}"
            elif agg_index == 3:  # COUNT
                return str(len(values))

            return str(values[0])
        except Exception as e:
            logger.error(f"Error processing coordinates: {e}")
            return "Error processing answer"
