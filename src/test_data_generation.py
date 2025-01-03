import logging
from prepare_training_data import (
    ConversationTemplate,
    CONVERSATION_FLOWS,
    generate_dynamic_response,
    create_natural_conversation,
    create_enhanced_dataset
)
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_response_generation():
    """Test dynamic response generation with templates"""
    context = {
        "market_context": "markets are showing increased volatility",
        "additional_info": "diversification becomes especially important"
    }
    
    template = "Based on current conditions, {explanation}. Given that {market_context}, {allocation} might be appropriate."
    
    # Generate multiple responses to check variation
    responses = [
        generate_dynamic_response(template, context)
        for _ in range(3)
    ]
    
    logger.info("Testing dynamic response generation:")
    for i, response in enumerate(responses, 1):
        logger.info(f"Response {i}: {response}")
        assert isinstance(response, str)
        assert len(response) > 0
        assert "{" not in response  # Ensure all placeholders were replaced
    
    return all(r != responses[0] for r in responses[1:])  # Check for variation

def test_natural_conversation_flow():
    """Test creation of natural conversation flows"""
    context = {
        "market_context": "markets are experiencing high volatility",
        "additional_info": "focus on risk management is crucial"
    }
    
    for flow_type in CONVERSATION_FLOWS.keys():
        conversation = create_natural_conversation(flow_type, context)
        
        logger.info(f"\nTesting {flow_type} conversation flow:")
        for i, turn in enumerate(conversation):
            logger.info(f"Turn {i+1}:")
            logger.info(f"Question: {turn['free_messages'][0]}")
            logger.info(f"Response: {turn['guided_messages'][0]}")
            
            # Verify structure
            assert isinstance(turn['free_messages'], list)
            assert isinstance(turn['guided_messages'], list)
            assert len(turn['free_messages']) == 1
            assert len(turn['guided_messages']) == 1
            
            if i > 0:  # Check context for followup questions
                assert turn['previous_utterance']
    
    return True

def test_dataset_generation():
    """Test the complete dataset generation pipeline"""
    test_output = Path("test_data.json")
    
    try:
        # Generate a small test dataset
        data = create_enhanced_dataset(
            str(test_output),
            max_samples=100,
            include_market_context=True,
            max_conversation_turns=2
        )
        
        # Verify the output
        logger.info("\nTesting dataset generation:")
        logger.info(f"Generated {len(data)} samples")
        
        # Check a few random samples
        import random
        samples = random.sample(data, min(5, len(data)))
        
        for i, sample in enumerate(samples):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Question: {sample['free_messages'][0]}")
            logger.info(f"Response: {sample['guided_messages'][0]}")
            
            # Verify required fields
            assert 'personas' in sample
            assert 'free_messages' in sample
            assert 'guided_messages' in sample
            
        return True
        
    finally:
        # Cleanup test file
        if test_output.exists():
            test_output.unlink()

def main():
    """Run all tests"""
    try:
        logger.info("Starting data generation tests...")
        
        # Run tests
        dynamic_response_ok = test_dynamic_response_generation()
        conversation_flow_ok = test_natural_conversation_flow()
        dataset_generation_ok = test_dataset_generation()
        
        # Report results
        logger.info("\nTest Results:")
        logger.info(f"Dynamic Response Generation: {'✓' if dynamic_response_ok else '✗'}")
        logger.info(f"Natural Conversation Flow: {'✓' if conversation_flow_ok else '✗'}")
        logger.info(f"Dataset Generation: {'✓' if dataset_generation_ok else '✗'}")
        
        if all([dynamic_response_ok, conversation_flow_ok, dataset_generation_ok]):
            logger.info("\nAll tests passed successfully! ✓")
        else:
            logger.error("\nSome tests failed! ✗")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
