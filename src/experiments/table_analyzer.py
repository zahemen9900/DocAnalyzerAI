import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import logging
import sys
sys.path.append('../src')

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
            # Load CSV with index column and reset index to make year a regular column
            df = pd.read_csv(filepath, index_col='Unnamed: 0').reset_index()
            df = df.rename(columns={'index': 'Year'})  # Rename index column to Year
            
            # Convert all numeric values to formatted strings
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].apply(lambda x: f"{x:,}")
            
            print(df.head())
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def query_table(self, table: pd.DataFrame, query: str) -> str:
        """Query the table using TAPAS model"""
        try:
            # Reset index if it exists to ensure proper coordinate handling
            if isinstance(table.index, pd.RangeIndex):
                table_for_query = table.copy()
            else:
                table_for_query = table.reset_index()
            
            # Convert DataFrame to string types
            table_for_query = table_for_query.astype(str)
            
            # Clean up the table for better querying
            table_for_query = table_for_query.replace('nan', '')
            
            # Encode the question and table
            inputs = self.tokenizer(
                table=table_for_query,
                queries=[query],
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            # Set model to evaluation mode and process
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(**inputs)
                logits = outputs.logits.detach()
                logits_agg = outputs.logits_aggregation.detach()
                
                predicted_answer_coords = self.tokenizer.convert_logits_to_predictions(
                    inputs,
                    logits,
                    logits_agg
                )
            
            coords, agg = predicted_answer_coords
            
            if coords and coords[0]:
                answer = self._get_answer_from_coords(table_for_query, coords[0], agg[0])
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

    def preprocess_query(self, query: str) -> str:
        """Clean and standardize queries"""
        # Handle common financial terms/aliases
        query = query.lower()
        replacements = {
            "revenue": "net sales",
            "profit": "net income",
            "earnings": "net income",
            "r&d": "research and development",
            "capex": "payments for acquisition of property, plant and equipment"
        }
        for old, new in replacements.items():
            query = query.replace(old, new)
        return query

    def calculate_growth_rate(self, table: pd.DataFrame, metric: str) -> dict:
        """Calculate year-over-year growth rates"""
        try:
            # Ensure we're working with the year column
            if 'Year' not in table.columns:
                table = table.reset_index()
                table = table.rename(columns={'index': 'Year'})
            
            # Convert string values back to numeric for calculation
            values = pd.to_numeric(table[metric].str.replace(',', ''))
            growth_rates = values.pct_change() * 100
            
            return {
                str(year): f"{rate:.2f}%" 
                for year, rate in zip(table['Year'], growth_rates)
            }
        except Exception as e:
            logger.error(f"Error calculating growth rate: {e}")
            return {}
