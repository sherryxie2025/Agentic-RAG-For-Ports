# src/api/server.py
"""
FastAPI Server for AI Port Decision-Support System
Provides REST API endpoints for all port operations and LangGraph workflows

Provides REST API endpoints for all port operations and LangGraph workflows
Handles:
1) Simple RAG queries
2) LangGraph workflow queries
3) What-if scenario analysis
4) Document similarity search
5) Comprehensive system status
6) Intent classification
7) Decision node thresholds

"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports
from ..graph.port_graph import PortWorkflowGraph
from ..routers.intent_router import IntentRouter
from ..routers.decision_nodes import DecisionNodes
from ..port_information_retrieval import PortInformationRetrieval
from ..utils.schemas import (
    QueryRequest, QueryResponse, WorkflowRequest, WorkflowResponse,
    WhatIfRequest, WhatIfResponse, SearchRequest, SearchResponse,
    HealthResponse, SystemStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Port Decision-Support System",
    description="Intelligent decision support for port operations using LangChain and LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (will be initialized in startup)
port_retrieval: Optional[PortInformationRetrieval] = None
workflow_graph: Optional[PortWorkflowGraph] = None
intent_router: Optional[IntentRouter] = None
decision_nodes: Optional[DecisionNodes] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the system components on startup"""
    global port_retrieval, workflow_graph, intent_router, decision_nodes
    
    try:
        logger.info("Starting AI Port Decision-Support System...")
        
        # Initialize core components
        port_retrieval = PortInformationRetrieval(
            openai_api_key="your-api-key",  # Will be loaded from environment
            chroma_persist_directory="./storage/chroma",
            collection_name="port_documents"
        )
        
        intent_router = IntentRouter()
        decision_nodes = DecisionNodes()
        
        workflow_graph = PortWorkflowGraph(
            port_retrieval=port_retrieval,
            enable_checkpoints=True
        )
        
        logger.info("System startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "AI Port Decision-Support System API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check system components
        components_status = {
            "port_retrieval": port_retrieval is not None,
            "workflow_graph": workflow_graph is not None,
            "intent_router": intent_router is not None,
            "decision_nodes": decision_nodes is not None
        }
        
        all_healthy = all(components_status.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            timestamp=datetime.now().isoformat(),
            components=components_status,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            components={},
            version="1.0.0",
            error=str(e)
        )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Simple RAG query endpoint for basic questions.
    Uses the port information retrieval system directly.
    """
    try:
        if not port_retrieval:
            raise HTTPException(status_code=503, detail="Port retrieval system not initialized")
        
        logger.info(f"Processing simple query: {request.query[:50]}...")
        
        # Process the query
        result = await port_retrieval.query_port_information(
            question=request.query,
            document_types=request.document_types,
            max_documents=request.max_documents or 5
        )
        
        return QueryResponse(
            query=request.query,
            answer=result.get("answer", "No answer available"),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.5),
            timestamp=datetime.now().isoformat(),
            processing_time=result.get("processing_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/ask_graph", response_model=WorkflowResponse)
async def ask_with_workflow(request: WorkflowRequest):
    """
    Advanced query endpoint using LangGraph workflow.
    Provides parallel analysis and decision gates.
    """
    try:
        if not workflow_graph:
            raise HTTPException(status_code=503, detail="Workflow graph not initialized")
        
        logger.info(f"Processing workflow query: {request.query[:50]}...")
        
        # Execute the workflow
        result = await workflow_graph.execute_workflow(
            query=request.query,
            user_id=request.user_id,
            port_config=request.port_config
        )
        
        return WorkflowResponse(
            query=request.query,
            intent=result.get("intent", "general"),
            confidence=result.get("confidence_score", 0.5),
            recommendation=result.get("final_recommendation", "No recommendation available"),
            alternative_scenarios=result.get("alternative_scenarios", []),
            sources=result.get("sources", []),
            decision_gates={
                "freshness": result.get("freshness_check", True),
                "confidence": result.get("confidence_check", True),
                "compliance": result.get("compliance_check", True)
            },
            execution_time=result.get("execution_time", 0.0),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            workflow_id=result.get("workflow_id", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Error processing workflow query: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow processing failed: {str(e)}")


@app.post("/what-if", response_model=WhatIfResponse)
async def what_if_analysis(request: WhatIfRequest):
    """
    What-if scenario analysis endpoint.
    Executes multiple scenarios in parallel using LangGraph.
    """
    try:
        if not workflow_graph:
            raise HTTPException(status_code=503, detail="Workflow graph not initialized")
        
        logger.info(f"Processing what-if analysis with {len(request.scenarios)} scenarios")
        
        # Execute what-if scenarios
        result = await workflow_graph.execute_what_if_scenario(
            base_query=request.base_query,
            scenario_variations=request.scenarios,
            user_id=request.user_id,
            port_config=request.port_config
        )
        
        return WhatIfResponse(
            base_query=request.base_query,
            scenarios=result.get("scenarios", {}),
            total_scenarios=result.get("total_scenarios", 0),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            execution_time=result.get("execution_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error processing what-if analysis: {e}")
        raise HTTPException(status_code=500, detail=f"What-if analysis failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Direct document similarity search endpoint.
    Returns similar documents without generating answers.
    """
    try:
        if not port_retrieval:
            raise HTTPException(status_code=503, detail="Port retrieval system not initialized")
        
        logger.info(f"Searching documents for: {request.query[:50]}...")
        
        # Search for similar documents
        documents = await port_retrieval.search_similar_documents(
            query=request.query,
            document_type=request.document_type,
            k=request.max_results or 10
        )
        
        return SearchResponse(
            query=request.query,
            documents=documents,
            total_found=len(documents),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")


@app.get("/summary", response_model=Dict[str, Any])
async def get_document_summary(document_type: Optional[str] = None):
    """
    Get summary of available documents in the system.
    """
    try:
        if not port_retrieval:
            raise HTTPException(status_code=503, detail="Port retrieval system not initialized")
        
        logger.info("Generating document summary")
        
        # Get document summary
        summary = await port_retrieval.get_document_summary(document_type)
        
        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@app.get("/intent/classify")
async def classify_intent(query: str, port_config: Optional[Dict[str, Any]] = None):
    """
    Classify a query intent without processing.
    Useful for debugging and testing intent classification.
    """
    try:
        if not intent_router:
            raise HTTPException(status_code=503, detail="Intent router not initialized")
        
        logger.info(f"Classifying intent for: {query[:50]}...")
        
        # Classify the intent
        result = await intent_router.classify_query(query, port_config)
        
        return {
            "query": query,
            "classification": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        raise HTTPException(status_code=500, detail=f"Intent classification failed: {str(e)}")


@app.get("/intent/statistics")
async def get_intent_statistics():
    """Get statistics about intent classification patterns"""
    try:
        if not intent_router:
            raise HTTPException(status_code=503, detail="Intent router not initialized")
        
        stats = intent_router.get_intent_statistics()
        
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting intent statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")


@app.get("/decision/thresholds")
async def get_decision_thresholds():
    """Get current decision node thresholds"""
    try:
        if not decision_nodes:
            raise HTTPException(status_code=503, detail="Decision nodes not initialized")
        
        thresholds = decision_nodes.get_threshold_status()
        
        return {
            "thresholds": thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting decision thresholds: {e}")
        raise HTTPException(status_code=500, detail=f"Threshold retrieval failed: {str(e)}")


@app.post("/decision/thresholds")
async def update_decision_thresholds(
    freshness_threshold_hours: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
    compliance_strict_mode: Optional[bool] = None
):
    """Update decision node thresholds"""
    try:
        if not decision_nodes:
            raise HTTPException(status_code=503, detail="Decision nodes not initialized")
        
        decision_nodes.update_thresholds(
            freshness_threshold_hours=freshness_threshold_hours,
            confidence_threshold=confidence_threshold,
            compliance_strict_mode=compliance_strict_mode
        )
        
        return {
            "message": "Thresholds updated successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating thresholds: {e}")
        raise HTTPException(status_code=500, detail=f"Threshold update failed: {str(e)}")


@app.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Check all components
        components = {
            "port_retrieval": port_retrieval is not None,
            "workflow_graph": workflow_graph is not None,
            "intent_router": intent_router is not None,
            "decision_nodes": decision_nodes is not None
        }
        
        # Get document summary if available
        document_summary = None
        if port_retrieval:
            try:
                document_summary = await port_retrieval.get_document_summary()
            except Exception as e:
                logger.warning(f"Could not get document summary: {e}")
        
        # Get intent statistics if available
        intent_stats = None
        if intent_router:
            try:
                intent_stats = intent_router.get_intent_statistics()
            except Exception as e:
                logger.warning(f"Could not get intent statistics: {e}")
        
        return SystemStatus(
            status="operational" if all(components.values()) else "degraded",
            components=components,
            document_summary=document_summary,
            intent_statistics=intent_stats,
            timestamp=datetime.now().isoformat(),
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
