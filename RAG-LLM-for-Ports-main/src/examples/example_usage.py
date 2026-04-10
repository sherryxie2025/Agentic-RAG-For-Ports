# src/examples/example_usage.py
"""
End-to-End Example Usage for AI Port Decision-Support System
Demonstrates complete workflow from document ingestion to intelligent querying
"""

# Sample example - 

"""
Step1 Question Asked ->  User asks: "How should I handle a berth conflict between two vessels?"
Step2 Intent Classification -> Intent Router analyzes: "berth conflict" → Intent: "berthing" (Priority: 2)
Step3 Document Retrieval -> Port Information Retrieval searches ChromaDB for relevant documents about berthing

Step4 LangGraph Workflow -> 

Port Graph runs parallel analysis:
├── Safety Analysis: "What are the safety implications?"
├── Efficiency Analysis: "How can we optimize this?"
├── Cost Analysis: "What are the cost implications?"
└── Risk Analysis: "What risks are involved?"

Step5 Decision Gates -> Decision Nodes checks:
├── Freshness Gate: "Is this information recent?" ✅
├── Confidence Gate: "Are we confident in these documents?" ✅
└── Compliance Gate: "Do recommendations follow safety rules?" ✅

Step6 Synthesize Results -> 
System combines all analysis results into:
- Final recommendation
- Alternative scenarios
- Source citations
- Confidence scores

Return response - User gets: "Based on port protocols, here are 3 approaches to handle the conflict..."

"""

import asyncio
import os
import logging
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Local imports
from ..port_information_retrieval import PortInformationRetrieval
from ..graph.port_graph import PortWorkflowGraph
from ..routers.intent_router import IntentRouter
from ..routers.decision_nodes import DecisionNodes
from ..utils.loaders import load_and_process_documents
from ..utils.redact import create_redactor
from ..utils.schemas import PortConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortSystemDemo:
    """
    Complete demonstration of the AI Port Decision-Support System.
    Shows end-to-end workflow from setup to intelligent querying.
    """
    
    def __init__(self):
        """Initialize the demo system"""
        self.port_retrieval = None
        self.workflow_graph = None
        self.intent_router = None
        self.decision_nodes = None
        self.redactor = None
        
        logger.info("Port System Demo initialized")
    
    async def setup_system(self, openai_api_key: str):
        """
        Set up the complete system with all components.
        
        Args:
            openai_api_key: OpenAI API key for LLM access
        """
        try:
            logger.info("Setting up AI Port Decision-Support System...")
            
            # Initialize core components
            self.port_retrieval = PortInformationRetrieval(
                # openai_api_key=openai_api_key,
                chroma_persist_directory="./storage/chroma",
                collection_name="port_documents",

                enable_answer_generation=False,
                enable_llm_in_retrieval=False
            )
            
            self.intent_router = IntentRouter()
            self.decision_nodes = DecisionNodes()
            self.redactor = create_redactor()
            
            self.workflow_graph = PortWorkflowGraph(
                port_retrieval=self.port_retrieval,
                enable_checkpoints=True
            )
            
            logger.info("✅ System setup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ System setup failed: {e}")
            raise
    
    async def ingest_sample_documents(self, documents_dir: str = "./data/sample_port_documents"):
        """
        Ingest sample documents for demonstration.
        
        Args:
            documents_dir: Directory containing sample documents
        """
        try:
            logger.info(f"📁 Ingesting documents from: {documents_dir}")
            
            # Check if documents directory exists
            if not os.path.exists(documents_dir):
                logger.warning(f"⚠️  Documents directory '{documents_dir}' not found")
                logger.info("Creating sample documents for demonstration...")
                await self._create_sample_documents(documents_dir)
            
            # Load and process documents
            documents = load_and_process_documents(
                pdf_directory=documents_dir,
                chunk_size=1000,
                chunk_overlap=200,
                multilingual=True,
                port_processing=True
            )
            
            if not documents:
                logger.warning("⚠️  No documents were processed")
                return False
            
            # Apply redaction
            logger.info("🔒 Applying data redaction...")
            redacted_docs = self.redactor.redact_documents(documents)
            
            # Add to vector store
            # success = self.port_retrieval.add_documents_to_vector_store(redacted_docs)
            success = await self.port_retrieval.add_documents_to_vector_store(redacted_docs)
            
            if success:
                logger.info(f"✅ Successfully ingested {len(redacted_docs)} document chunks")
                return True
            else:
                logger.error("❌ Failed to add documents to vector store")
                return False
                
        except Exception as e:
            logger.error(f"❌ Document ingestion failed: {e}")
            return False
    
    async def _create_sample_documents(self, documents_dir: str):
        """Create sample documents for demonstration"""
        try:
            os.makedirs(documents_dir, exist_ok=True)
            
            # Create sample document content
            sample_docs = {
                "safety_protocols.pdf": """
                PORT SAFETY PROTOCOLS
                
                Emergency Procedures:
                1. In case of fire, contact emergency services at 911
                2. Evacuation routes are marked with green signs
                3. All personnel must wear safety equipment
                
                Vessel Safety:
                - IMO: 1234567
                - MMSI: 123456789
                - Call Sign: ABC123
                
                Contact: safety@port.com
                """,
                
                "berth_schedule.pdf": """
                BERTH ALLOCATION SCHEDULE
                
                Berth A1: Container Vessel "Ocean Star"
                - Arrival: 2024-01-15 08:00
                - Departure: 2024-01-15 18:00
                - Cargo: 500 containers
                
                Berth B2: Bulk Carrier "Cargo Master"
                - Arrival: 2024-01-15 10:00
                - Departure: 2024-01-16 06:00
                - Cargo: Coal, $2,500,000 value
                
                Weather Advisory: High winds expected
                """,
                
                "weather_report.pdf": """
                PORT WEATHER REPORT
                
                Current Conditions:
                - Wind: 25 knots from NW
                - Visibility: 2 nautical miles
                - Sea State: Moderate
                
                Forecast:
                - Storm warning in effect
                - Expected wind gusts up to 40 knots
                - Recommend delaying operations
                
                Tide Information:
                - High tide: 14:30
                - Low tide: 08:15
                """
            }
            
            # Write sample documents
            for filename, content in sample_docs.items():
                filepath = os.path.join(documents_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
            
            logger.info(f"✅ Created {len(sample_docs)} sample documents")
            
        except Exception as e:
            logger.error(f"❌ Error creating sample documents: {e}")
    
    async def demonstrate_simple_queries(self):
        """Demonstrate simple RAG queries"""
        try:
            logger.info("🔍 Demonstrating Simple RAG Queries")
            print("\n" + "="*60)
            print("SIMPLE RAG QUERIES DEMONSTRATION")
            print("="*60)
            
            sample_queries = [
                "What are the safety protocols for vessel berthing?",
                "What is the current weather forecast?",
                "Which vessels are scheduled to arrive today?",
                "What should I do in case of an emergency?"
            ]
            
            for i, query in enumerate(sample_queries, 1):
                print(f"\n📋 Query {i}: {query}")
                print("-" * 50)
                
                result = await self.port_retrieval.query_port_information(
                    question=query,
                    max_documents=3
                )
                
                print(f"✅ Answer: {result['answer'][:200]}...")
                print(f"📚 Sources: {len(result['sources'])} documents")
                
                # Show source details
                for j, source in enumerate(result['sources'][:2], 1):
                    print(f"   {j}. {source['source']} ({source['document_type']})")
                
        except Exception as e:
            logger.error(f"❌ Simple queries demonstration failed: {e}")


    # 新增 Retrieval Debug Demo
    async def demonstrate_retrieval(self):

        print("\n" + "="*60)
        print("RETRIEVAL PIPELINE DEBUG")
        print("="*60)

        query = "berth conflict between vessels"


        docs = await self.port_retrieval.search_similar_documents(query, k=5)

        print(f"\nRetrieved {len(docs)} documents")

        for i, doc in enumerate(docs, 1):

            print("\n-----------------------------")
            print(f"Result {i}")
            print("Source:", doc.get("source"))
            print("Type:", doc.get("document_type"))
            print("Content:", doc.get("content")[:200])

    
    async def demonstrate_workflow_queries(self):
        """Demonstrate LangGraph workflow queries"""
        try:
            logger.info("🧠 Demonstrating LangGraph Workflow Queries")
            print("\n" + "="*60)
            print("LANGGRAPH WORKFLOW DEMONSTRATION")
            print("="*60)
            
            workflow_queries = [
                "How should I handle a berth conflict between two vessels?",
                "What are the safety implications of operating in high winds?",
                "How can I optimize cargo operations for efficiency?"
            ]
            
            for i, query in enumerate(workflow_queries, 1):
                print(f"\n🎯 Workflow Query {i}: {query}")
                print("-" * 50)
                
                result = await self.workflow_graph.execute_workflow(
                    query=query,
                    user_id=f"demo_user_{i}",
                    port_config={
                        "port_type": "container",
                        "has_weather_station": True
                    }
                )
                
                print(f"🎯 Intent: {result.get('intent', 'Unknown')}")
                print(f"📊 Confidence: {result.get('confidence_score', 0):.2f}")
                print(f"💡 Recommendation: {result.get('final_recommendation', 'No recommendation')[:200]}...")
                
                # Show decision gates
                gates = {
                    "Freshness": result.get('freshness_check', False),
                    "Confidence": result.get('confidence_check', False),
                    "Compliance": result.get('compliance_check', False)
                }
                print(f"🚦 Decision Gates: {gates}")
                
                # Show alternative scenarios
                scenarios = result.get('alternative_scenarios', [])
                if scenarios:
                    print(f"🔄 Alternative Scenarios: {len(scenarios)}")
                    for j, scenario in enumerate(scenarios[:2], 1):
                        print(f"   {j}. {scenario.get('scenario_name', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"❌ Workflow queries demonstration failed: {e}")
    
    async def demonstrate_what_if_scenarios(self):
        """Demonstrate what-if scenario analysis"""
        try:
            logger.info("🔄 Demonstrating What-If Scenario Analysis")
            print("\n" + "="*60)
            print("WHAT-IF SCENARIO ANALYSIS DEMONSTRATION")
            print("="*60)
            
            base_query = "How should I handle vessel berthing operations?"
            scenarios = [
                "in normal weather conditions",
                "during a storm warning",
                "with equipment breakdown",
                "with high cargo priority"
            ]
            
            print(f"🎯 Base Query: {base_query}")
            print(f"🔄 Scenarios: {len(scenarios)} variations")
            print("-" * 50)
            
            result = await self.workflow_graph.execute_what_if_scenario(
                base_query=base_query,
                scenario_variations=scenarios,
                user_id="demo_user_scenarios"
            )
            
            print(f"✅ Analysis completed in {result.get('execution_time', 0):.2f} seconds")
            
            # Show scenario results
            scenario_results = result.get('scenarios', {})
            for scenario_key, scenario_data in scenario_results.items():
                if 'error' not in scenario_data:
                    variation = scenario_data.get('variation', 'Unknown')
                    print(f"\n📋 Scenario: {variation}")
                    print(f"   Intent: {scenario_data.get('result', {}).get('intent', 'Unknown')}")
                    print(f"   Confidence: {scenario_data.get('result', {}).get('confidence_score', 0):.2f}")
                
        except Exception as e:
            logger.error(f"❌ What-if scenarios demonstration failed: {e}")
    
    async def demonstrate_intent_classification(self):
        """Demonstrate intent classification"""
        try:
            logger.info("🎯 Demonstrating Intent Classification")
            print("\n" + "="*60)
            print("INTENT CLASSIFICATION DEMONSTRATION")
            print("="*60)
            
            test_queries = [
                "What's the weather forecast for tomorrow?",
                "How do I handle a safety emergency?",
                "Which berth should I assign to the incoming vessel?",
                "What are the cargo handling procedures?",
                "Is the crane equipment working properly?",
                "What regulations do I need to follow?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n❓ Query {i}: {query}")
                
                result = await self.intent_router.classify_query(query)
                
                print(f"🎯 Intent: {result['intent']}")
                print(f"📊 Confidence: {result['confidence']:.2f}")
                print(f"🛤️  Routing: {result['routing_path']}")
                print(f"⭐ Priority: {result['priority']}")
                
                if result.get('sub_intent'):
                    print(f"🔍 Sub-intent: {result['sub_intent']}")
                
        except Exception as e:
            logger.error(f"❌ Intent classification demonstration failed: {e}")
    
    async def demonstrate_decision_gates(self):
        """Demonstrate decision gate functionality"""
        try:
            logger.info("🚦 Demonstrating Decision Gates")
            print("\n" + "="*60)
            print("DECISION GATES DEMONSTRATION")
            print("="*60)
            
            # Get current thresholds
            thresholds = self.decision_nodes.get_threshold_status()
            print(f"📊 Current Thresholds:")
            print(f"   Freshness: {thresholds['freshness_threshold_hours']} hours")
            print(f"   Confidence: {thresholds['confidence_threshold']}")
            print(f"   Compliance Strict: {thresholds['compliance_strict_mode']}")
            
            # Test decision gates with sample documents
            sample_docs = [
                {
                    "content": "Emergency procedures updated 2024-01-15. Contact safety@port.com",
                    "source": "safety_protocols.pdf",
                    "document_type": "safety_protocols",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "content": "Weather forecast from 2023-12-01. Storm warning issued.",
                    "source": "weather_report.pdf", 
                    "document_type": "weather_reports",
                    "timestamp": "2023-12-01T00:00:00"
                }
            ]
            
            test_query = "What are the emergency procedures?"
            
            print(f"\n🧪 Testing Decision Gates with Query: {test_query}")
            print("-" * 50)
            
            # Test freshness gate
            freshness_result = await self.decision_nodes.check_freshness(
                documents=sample_docs,
                query=test_query
            )
            print(f"🕒 Freshness Gate: {'✅ PASS' if freshness_result.passed else '❌ FAIL'}")
            print(f"   Score: {freshness_result.score:.2f}")
            print(f"   Reason: {freshness_result.reason}")
            
            # Test confidence gate
            confidence_result = await self.decision_nodes.check_confidence(
                documents=sample_docs,
                query=test_query
            )
            print(f"📊 Confidence Gate: {'✅ PASS' if confidence_result.passed else '❌ FAIL'}")
            print(f"   Score: {confidence_result.score:.2f}")
            print(f"   Reason: {confidence_result.reason}")
            
            # Test compliance gate
            compliance_result = await self.decision_nodes.check_compliance(
                documents=sample_docs,
                query=test_query,
                intent="safety"
            )
            print(f"🛡️  Compliance Gate: {'✅ PASS' if compliance_result.passed else '❌ FAIL'}")
            print(f"   Score: {compliance_result.score:.2f}")
            print(f"   Reason: {compliance_result.reason}")
                
        except Exception as e:
            logger.error(f"❌ Decision gates demonstration failed: {e}")
    
    async def demonstrate_redaction(self):
        """Demonstrate data redaction functionality"""
        try:
            logger.info("🔒 Demonstrating Data Redaction")
            print("\n" + "="*60)
            print("DATA REDACTION DEMONSTRATION")
            print("="*60)
            
            # Sample text with sensitive information
            sample_text = """
            VESSEL INFORMATION:
            Vessel Name: Ocean Star
            IMO: 1234567
            MMSI: 123456789
            Call Sign: ABC123
            
            CARGO DETAILS:
            Container: ABCD1234567
            Cargo Value: $2,500,000
            Booking Ref: BK123456
            
            CONTACT INFORMATION:
            Captain: John Smith
            Email: captain@oceanstar.com
            Phone: 555-123-4567
            
            EMERGENCY CONTACT:
            Port Authority: PA123
            Customs Decl: CD789012
            """
            
            print("📄 Original Text:")
            print(sample_text)
            print("\n🔒 After Redaction:")
            
            # Apply redaction
            redacted_text, redaction_counts = self.redactor.redact_text(sample_text)
            print(redacted_text)
            
            print(f"\n📊 Redaction Statistics:")
            for rule_name, count in redaction_counts.items():
                print(f"   {rule_name}: {count} matches")
            
            print(f"   Total redactions: {sum(redaction_counts.values())}")
                
        except Exception as e:
            logger.error(f"❌ Redaction demonstration failed: {e}")
    
    async def show_system_status(self):
        """Show comprehensive system status"""
        try:
            logger.info("📊 Showing System Status")
            print("\n" + "="*60)
            print("SYSTEM STATUS OVERVIEW")
            print("="*60)
            
            # Document summary
            if self.port_retrieval:
                doc_summary = await self.port_retrieval.get_document_summary()
                print(f"📚 Documents in System:")
                print(f"   Total: {doc_summary.get('total_documents', 0)}")
                
                doc_types = doc_summary.get('document_types', {})
                for doc_type, count in doc_types.items():
                    print(f"   {doc_type}: {count}")
            
            # Intent statistics
            if self.intent_router:
                intent_stats = self.intent_router.get_intent_statistics()
                print(f"\n🎯 Intent Classification:")
                print(f"   Total intents: {intent_stats['total_intents']}")
                print(f"   Priority levels: {intent_stats['priority_levels']}")
            
            # Decision thresholds
            if self.decision_nodes:
                thresholds = self.decision_nodes.get_threshold_status()
                print(f"\n🚦 Decision Thresholds:")
                print(f"   Freshness: {thresholds['freshness_threshold_hours']} hours")
                print(f"   Confidence: {thresholds['confidence_threshold']}")
                print(f"   Compliance Strict: {thresholds['compliance_strict_mode']}")
            
            # Redaction statistics
            if self.redactor:
                redaction_stats = self.redactor.get_redaction_statistics()
                print(f"\n🔒 Redaction Rules:")
                print(f"   Total rules: {redaction_stats['total_rules']}")
                print(f"   Enabled rules: {redaction_stats['enabled_rules']}")
                
                categories = redaction_stats['rule_categories']
                print(f"   Maritime: {categories['maritime']}")
                print(f"   Financial: {categories['financial']}")
                print(f"   PII: {categories['pii']}")
                print(f"   Port-specific: {categories['port_specific']}")
                
        except Exception as e:
            logger.error(f"❌ System status display failed: {e}")
    
    async def run_complete_demo(self, openai_api_key: str):
        """
        Run the complete demonstration of the AI Port Decision-Support System.
        
        Args:
            openai_api_key: OpenAI API key for LLM access
        """
        try:
            print("🚢 AI PORT DECISION-SUPPORT SYSTEM DEMO")
            print("=" * 60)
            print("This demo showcases the complete workflow from document")
            print("ingestion to intelligent query processing with LangGraph.")
            print("=" * 60)
            
            # Setup system
            await self.setup_system(openai_api_key)
            
            # Ingest documents
            await self.ingest_sample_documents()

            await self.demonstrate_retrieval()   # 新增
            
            # Run demonstrations
            await self.demonstrate_simple_queries()
            await self.demonstrate_workflow_queries()
            await self.demonstrate_what_if_scenarios()
            await self.demonstrate_intent_classification()
            await self.demonstrate_decision_gates()
            await self.demonstrate_redaction()
            await self.show_system_status()
            
            print("\n" + "="*60)
            print("✅ DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("The AI Port Decision-Support System is ready for production use.")
            print("Key features demonstrated:")
            print("• Document ingestion with multilingual support")
            print("• Intelligent intent classification")
            print("• LangGraph workflow orchestration")
            print("• Parallel branching for what-if scenarios")
            print("• Decision gates for information validation")
            print("• Advanced data redaction for security")
            print("• Comprehensive API endpoints")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            print(f"\n❌ Demo failed: {e}")
            print("Please check your OpenAI API key and try again.")


async def main():
    """Main function to run the demo"""
    
    # Get OpenAI API key
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_key = "dummy"
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create and run demo
    demo = PortSystemDemo()
    await demo.run_complete_demo(openai_api_key)


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())