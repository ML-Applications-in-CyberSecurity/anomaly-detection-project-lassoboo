import socket
import json
import pandas as pd
import joblib
import requests
import os
import csv
import time
import statistics
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum
import logging

# Load environment variables from .env file
load_dotenv()

# =============================== Configuration ================================================ #
HOST = 'localhost'
PORT = 9999

# API Configuration from environment variables
TOGETHER_API_URL = os.getenv('TOGETHER_API_URL')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
MAX_TOKENS = int(os.getenv('MAX_TOKENS'))
TEMPERATURE = float(os.getenv('TEMPERATURE'))
DEFAULT_TEMPLATE = os.getenv('DEFAULT_TEMPLATE')
ENABLE_MULTI_TEMPLATE = os.getenv('ENABLE_MULTI_TEMPLATE', 'false').lower() == 'true'
SAVE_COMPARISON_REPORTS = os.getenv('SAVE_COMPARISON_REPORTS', 'true').lower() == 'true'

LOG_FILE = 'src/anomaly_log.csv'
PERFORMANCE_LOG = 'template_performance.json'
COMPARISON_REPORT_DIR = './reports/'

# Create reports directory if it doesn't exist
os.makedirs(COMPARISON_REPORT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================== Template System ====================================== #

class PromptStyle(Enum):
    """Different prompt styles for LLM analysis"""
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"
    CONTEXTUAL = "contextual"
    SEVERITY_FOCUSED = "severity_focused"


@dataclass
class PromptTemplate:
    """Class to hold different prompt templates"""
    name: str
    style: PromptStyle
    system_message: str
    user_template: str
    description: str
    expected_format: str


class LLMAnalyzer:
    """Enhanced class for anomaly analysis with multiple LLM prompt templates"""

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.templates = self._initialize_templates()
        self.results_log = []
        self.performance_metrics = {}
        self.template_cache = {}
        self.load_performance_history()

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize different prompt templates for LLM analysis"""

        templates = {}

        # 1. Technical Expert Template
        templates["technical_expert"] = PromptTemplate(
            name="Technical Expert",
            style=PromptStyle.TECHNICAL,
            system_message="""You are a senior cybersecurity engineer with 15+ years of experience in network security, 
            threat hunting, and incident response. You specialize in analyzing network traffic anomalies and have deep 
            expertise in attack patterns, malware behavior, and network forensics. Provide precise, technical analysis 
            with specific indicators of compromise (IoCs) and technical recommendations.""",

            user_template="""NETWORK ANOMALY ANALYSIS REQUEST

TRAFFIC CHARACTERISTICS:
- Source Port: {src_port}
- Destination Port: {dst_port}
- Packet Size: {packet_size} bytes
- Connection Duration: {duration_ms} ms
- Protocol: {protocol}

ANOMALY METRICS:
- ML Anomaly Score: {anomaly_score} (more negative = more anomalous)
- Detection Algorithm: Isolation Forest
- Classification: {anomaly_type}

ANALYSIS REQUIREMENTS:
1. TECHNICAL ASSESSMENT: Detailed technical analysis of the anomaly
2. IOC IDENTIFICATION: Specific indicators of compromise
3. ATTACK VECTOR: Most likely attack methodology
4. SEVERITY RATING: Critical/High/Medium/Low with justification
5. MITIGATION STEPS: Specific technical countermeasures
6. FORENSIC NOTES: Additional investigation recommendations

Provide detailed technical analysis suitable for security operations center (SOC) analysts.""",

            description="Detailed technical analysis for SOC teams",
            expected_format="Technical report with IoCs and specific recommendations"
        )

        # 2. Risk Assessment Template
        templates["risk_assessor"] = PromptTemplate(
            name="Risk Assessor",
            style=PromptStyle.STRUCTURED,
            system_message="""You are a cybersecurity risk analyst focused on business impact assessment. 
            Your role is to evaluate security threats from a risk management perspective, considering business 
            continuity, data protection, and organizational impact. Provide structured risk assessments.""",

            user_template="""SECURITY RISK EVALUATION

ANOMALY DATA:
• Network Source: Port {src_port}
• Network Destination: Port {dst_port}
• Data Volume: {packet_size} bytes
• Session Length: {duration_ms} milliseconds
• Protocol Type: {protocol}
• Risk Score: {anomaly_score}
• Threat Category: {anomaly_type}

REQUIRED RISK ASSESSMENT:

RISK LEVEL: [Critical/High/Medium/Low]
BUSINESS IMPACT: [Describe potential business consequences]
DATA AT RISK: [Types of data potentially compromised]
LIKELIHOOD: [Probability of successful attack]
FINANCIAL IMPACT: [Estimated cost of potential breach]
REGULATORY CONCERNS: [Compliance implications]
IMMEDIATE ACTIONS: [Priority response steps]
LONG-TERM STRATEGY: [Strategic security improvements]

Format response as structured risk assessment suitable for executive briefing.""",

            description="Business-focused risk assessment for management",
            expected_format="Structured risk analysis with business impact focus"
        )

        # 3. Incident Response Template
        templates["incident_responder"] = PromptTemplate(
            name="Incident Responder",
            style=PromptStyle.CONVERSATIONAL,
            system_message="""You are an experienced incident response coordinator leading a security team during 
            a potential security incident. Your communication style is clear, actionable, and focused on rapid 
            containment and resolution. Provide step-by-step incident response guidance.""",

            user_template="""POTENTIAL SECURITY INCIDENT - IMMEDIATE RESPONSE NEEDED

We've detected suspicious network activity with the following characteristics:
- Traffic originating from port {src_port}
- Communicating with port {dst_port}
- Unusual packet size: {packet_size} bytes
- Extended connection duration: {duration_ms} ms
- Protocol: {protocol}
- Anomaly confidence: {anomaly_score}
- Suspected threat type: {anomaly_type}

Team, I need your assessment for immediate incident response:

1. IMMEDIATE CONTAINMENT: What should we isolate right now?
2. EVIDENCE PRESERVATION: What logs/data should we secure?
3. INVESTIGATION PRIORITIES: What should we check first?
4. TEAM COORDINATION: Who needs to be notified?
5. TIMELINE: What's our response timeframe?
6. ESCALATION TRIGGERS: When do we escalate this further?

Please provide clear, actionable guidance for our incident response team.""",

            description="Actionable incident response coordination",
            expected_format="Step-by-step incident response plan"
        )

        # 4. Threat Intelligence Template
        templates["threat_intel"] = PromptTemplate(
            name="Threat Intelligence Analyst",
            style=PromptStyle.CONTEXTUAL,
            system_message="""You are a threat intelligence analyst with access to global threat feeds, 
            attack pattern databases, and adversary tactics, techniques, and procedures (TTPs). Analyze 
            network anomalies in the context of current threat landscape and known attack campaigns.""",

            user_template="""THREAT INTELLIGENCE CORRELATION REQUEST

OBSERVED ANOMALY:
Source Port: {src_port} | Destination Port: {dst_port} | Size: {packet_size}B | Duration: {duration_ms}ms | Protocol: {protocol}
Anomaly Score: {anomaly_score} | Classification: {anomaly_type}

INTELLIGENCE ANALYSIS REQUIRED:

THREAT LANDSCAPE CONTEXT:
- Does this match known threat actor TTPs?
- Any correlation with recent threat campaigns?
- Historical precedent for this pattern?

GEOPOLITICAL CONTEXT:
- Current threat actor activity levels
- Recent vulnerability exploitation trends
- Industry-specific targeting patterns

ATTACK CHAIN ANALYSIS:
- Where does this fit in the cyber kill chain?
- Likely next steps if this is malicious
- Associated infrastructure patterns

THREAT ATTRIBUTION:
- Possible threat actor groups
- Campaign similarities
- Confidence level in attribution

STRATEGIC IMPLICATIONS:
- Long-term threat trajectory
- Recommended threat hunting activities
- Intelligence collection priorities

Provide comprehensive threat intelligence analysis with confidence ratings.""",

            description="Threat intelligence and attribution analysis",
            expected_format="Intelligence briefing with threat actor context"
        )

        # 5. Executive Summary Template
        templates["executive_briefing"] = PromptTemplate(
            name="Executive Briefing",
            style=PromptStyle.SEVERITY_FOCUSED,
            system_message="""You are a Chief Information Security Officer (CISO) preparing executive briefings. 
            Your audience includes C-level executives who need clear, concise security updates without technical 
            jargon. Focus on business impact, decision points, and resource requirements.""",

            user_template="""EXECUTIVE SECURITY BRIEFING - ANOMALY DETECTED

A network security monitoring system has identified suspicious activity requiring executive awareness:

INCIDENT SUMMARY:
• Detection Time: {timestamp}
• Anomaly Severity: Based on score {anomaly_score}
• Traffic Pattern: Port {src_port} → Port {dst_port}
• Data Characteristics: {packet_size} bytes, {duration_ms}ms duration
• Protocol: {protocol}
• Threat Classification: {anomaly_type}

EXECUTIVE BRIEFING REQUIRED:

BUSINESS IMPACT SUMMARY (2-3 sentences)
FINANCIAL RISK ASSESSMENT (Budget implications)
TIME-SENSITIVE DECISIONS (What needs approval now?)
RESOURCE REQUIREMENTS (Team/budget needs)
BOARD REPORTING (Should this go to the board?)
FOLLOW-UP TIMELINE (Next update schedule)

Keep language business-focused, avoid technical jargon, and highlight decision points requiring executive input.""",

            description="Executive-level security briefing",
            expected_format="Business-focused summary for C-level executives"
        )

        return templates

    def analyze_with_template(self, data: Dict[str, Any], template_name: str,
                              anomaly_score: float, anomaly_type: str) -> Dict[str, Any]:
        """Analyze anomaly using specific template"""

        if template_name not in self.templates:
            logger.error(f"Template '{template_name}' not found")
            raise ValueError(f"Template '{template_name}' not found")

        template = self.templates[template_name]

        # Prepare template variables
        template_vars = {
            'src_port': data['src_port'],
            'dst_port': data['dst_port'],
            'packet_size': data['packet_size'],
            'duration_ms': data['duration_ms'],
            'protocol': data['protocol'],
            'anomaly_score': anomaly_score,
            'anomaly_type': anomaly_type,
            'timestamp': datetime.now().isoformat(),
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Format the user message
        user_message = template.user_template.format(**template_vars)

        # Make API call
        start_time = time.time()
        response = self._call_llm_api(template.system_message, user_message)
        end_time = time.time()
        response_time = end_time - start_time

        # Log results
        result = {
            'template_name': template_name,
            'template_style': template.style.value,
            'timestamp': datetime.now().isoformat(),
            'response_time': response_time,
            'data': data,
            'anomaly_score': anomaly_score,
            'anomaly_type': anomaly_type,
            'llm_response': response,
            'template_description': template.description
        }

        self.results_log.append(result)
        self.update_performance_metrics(template_name, response_time, len(response))

        return result

    def _call_llm_api(self, system_message: str, user_message: str) -> str:
        """Make API call to LLM with retry logic"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stop": ["<|eot_id|>"]
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content'].strip()

            except requests.exceptions.Timeout:
                logger.warning(f"API timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    return "⚠️ LLM analysis timeout - request took too long"
                time.sleep(2 ** attempt)  # Exponential backoff

            except requests.exceptions.RequestException as e:
                logger.error(f"API error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return f"⚠️ LLM API error: {str(e)}"
                time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return f"⚠️ Error generating LLM alert: {str(e)}"

        return "⚠️ Failed to generate analysis after multiple attempts"

    def compare_all_templates(self, data: Dict[str, Any], anomaly_score: float,
                              anomaly_type: str) -> Dict[str, Any]:
        """Run analysis with all templates and compare results"""

        logger.info("Running analysis with all prompt templates...")
        results = {}

        for template_name in self.templates.keys():
            logger.info(f"Processing with {template_name}...")
            try:
                result = self.analyze_with_template(data, template_name, anomaly_score, anomaly_type)
                results[template_name] = result
            except Exception as e:
                logger.error(f"Failed to process template {template_name}: {str(e)}")
                results[template_name] = {
                    'error': str(e),
                    'template_name': template_name,
                    'timestamp': datetime.now().isoformat()
                }

        # Generate comparison summary
        comparison = self._generate_comparison_summary(results)

        return {
            'individual_results': results,
            'comparison_summary': comparison,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary of all template results"""

        summary = {
            'response_times': {},
            'response_lengths': {},
            'severity_assessments': {},
            'success_count': 0,
            'error_count': 0
        }

        for template_name, result in results.items():
            if 'error' in result:
                summary['error_count'] += 1
                continue

            summary['success_count'] += 1

            # Response time analysis
            summary['response_times'][template_name] = result['response_time']

            # Response length analysis
            response_length = len(result['llm_response'].split())
            summary['response_lengths'][template_name] = response_length

            # Extract severity mentions
            response_lower = result['llm_response'].lower()
            if 'critical' in response_lower:
                summary['severity_assessments'][template_name] = 'Critical'
            elif 'high' in response_lower:
                summary['severity_assessments'][template_name] = 'High'
            elif 'medium' in response_lower:
                summary['severity_assessments'][template_name] = 'Medium'
            elif 'low' in response_lower:
                summary['severity_assessments'][template_name] = 'Low'
            else:
                summary['severity_assessments'][template_name] = 'Unspecified'

        return summary

    def update_performance_metrics(self, template_name: str, response_time: float, response_length: int):
        """Update performance metrics for templates"""

        if template_name not in self.performance_metrics:
            self.performance_metrics[template_name] = {
                'response_times': [],
                'response_lengths': [],
                'usage_count': 0,
                'error_count': 0
            }

        self.performance_metrics[template_name]['response_times'].append(response_time)
        self.performance_metrics[template_name]['response_lengths'].append(response_length)
        self.performance_metrics[template_name]['usage_count'] += 1

        # Keep only recent metrics (last 100)
        if len(self.performance_metrics[template_name]['response_times']) > 100:
            self.performance_metrics[template_name]['response_times'] = \
                self.performance_metrics[template_name]['response_times'][-100:]
            self.performance_metrics[template_name]['response_lengths'] = \
                self.performance_metrics[template_name]['response_lengths'][-100:]

    def get_best_template_for_context(self, context: Dict[str, Any]) -> str:
        """Select best template based on context and performance history"""

        # Context-based selection
        if context.get('severity') == 'critical':
            return 'incident_responder'
        elif context.get('audience') == 'executive':
            return 'executive_briefing'
        elif context.get('business_hours') and context.get('severity') in ['high', 'medium']:
            return 'risk_assessor'
        elif context.get('investigation_mode'):
            return 'threat_intel'
        elif not context.get('business_hours'):
            return 'technical_expert'
        else:
            # Use performance-based selection
            return self._select_by_performance()

    def _select_by_performance(self) -> str:
        """Select template based on performance metrics"""

        best_template = DEFAULT_TEMPLATE
        best_score = 0

        for template_name, metrics in self.performance_metrics.items():
            if metrics['usage_count'] < 3:  # Not enough data
                continue

            avg_response_time = statistics.mean(metrics['response_times'])
            avg_response_length = statistics.mean(metrics['response_lengths'])
            error_rate = metrics['error_count'] / metrics['usage_count']

            # Calculate performance score (lower response time and error rate is better)
            score = (1 / max(avg_response_time, 0.1)) * (1 - error_rate) * (avg_response_length / 1000)

            if score > best_score:
                best_score = score
                best_template = template_name

        return best_template

    def save_performance_metrics(self):
        """Save performance metrics to file"""

        try:
            with open(PERFORMANCE_LOG, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {str(e)}")

    def load_performance_history(self):
        """Load performance metrics from file"""

        try:
            if os.path.exists(PERFORMANCE_LOG):
                with open(PERFORMANCE_LOG, 'r') as f:
                    self.performance_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load performance metrics: {str(e)}")
            self.performance_metrics = {}


# =============================== Anomaly Detection ================================== #

class ContextAwareAnomalyDetector:
    """Enhanced anomaly detector with context awareness"""

    def __init__(self):
        self.llm_analyzer = None
        self.model = None
        self.analysis_history = []
        self.context_cache = {}
        self.load_model()
        self.initialize_llm()

    def load_model(self):
        """Load the trained anomaly detection model"""
        try:
            self.model = joblib.load("anomaly_model.joblib")
            logger.info("Anomaly detection model loaded successfully")
        except FileNotFoundError:
            logger.error("Error: anomaly_model.joblib not found. Please run train_model.ipynb first.")
            exit(1)

    def initialize_llm(self):
        """Initialize LLM analyzer"""
        if TOGETHER_API_KEY and TOGETHER_API_KEY != "your_together_ai_api_key_here":
            try:
                self.llm_analyzer = LLMAnalyzer(
                    api_url=TOGETHER_API_URL,
                    api_key=TOGETHER_API_KEY,
                    model_name=MODEL_NAME
                )
                logger.info("Enhanced LLM analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LLM analyzer: {str(e)}")
        else:
            logger.warning("LLM analyzer not initialized - API key not configured")

    def pre_process_data(self, data):
        """Convert incoming data to DataFrame and preprocess it for model prediction"""

        df = pd.DataFrame([data])

        # One-hot encode protocol column to match training
        df_encoded = pd.get_dummies(df, columns=['protocol'], drop_first=True)

        # Ensure all expected columns are present
        expected_columns = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reorder columns to match training data
        df_encoded = df_encoded[expected_columns]

        return df_encoded

    def detect_anomaly(self, data):
        """Use the trained model to detect if the data point is an anomaly"""

        try:
            processed_data = self.pre_process_data(data)

            # Predict: 1 = normal, -1 = anomaly
            prediction = self.model.predict(processed_data.values)[0]

            # Get anomaly score (confidence)
            anomaly_score = self.model.decision_function(processed_data.values)[0]

            is_anomaly = prediction == -1
            return is_anomaly, anomaly_score

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0

    def classify_anomaly_type(self, data):
        """Classify the type of anomaly based on data characteristics"""

        suspicious_ports = [1337, 9999, 6666, 31337, 12345, 54321]
        common_service_ports = [80, 443, 22, 8080]
        current_time = datetime.now()
        is_business_hours = 9 <= current_time.hour <= 17
        is_weekend = current_time.weekday() >= 5

        # Suspicious Source Port
        if data['src_port'] in suspicious_ports:
            return "SUSPICIOUS_SRC_PORT"

        # Suspicious Destination Port
        if data['dst_port'] in suspicious_ports:
            return "SUSPICIOUS_DST_PORT"

        # Large Packet Size (adjusted threshold based on protocol)
        packet_size_threshold = 2000 if data['protocol'] == "TCP" else 1000
        if data['packet_size'] > packet_size_threshold:
            return "LARGE_PACKET"

        # 4. Long Connection Duration (context-aware)
        duration_threshold = 1000 if is_business_hours else 1500
        if data['duration_ms'] > duration_threshold:
            return "LONG_DURATION"

        # Unknown or Unexpected Protocol
        if data['protocol'] not in ["TCP", "UDP", "ICMP"]:
            return "UNKNOWN_PROTOCOL"

        # Unusual Packet Size for Service Ports
        if data['src_port'] in common_service_ports:
            min_packet_size = 300 if data['src_port'] in [80, 443, 8080] else 100
            if data['packet_size'] < min_packet_size:
                return "UNUSUAL_PACKET_SIZE"

        # Potential Command-and-Control (C2) Communication
        if (data['src_port'] > 1024 or data['dst_port'] > 1024) and \
                data['duration_ms'] > (800 if is_business_hours else 1200) and \
                data['packet_size'] > 500:
            return "POTENTIAL_C2_COMMUNICATION"

        # Protocol-Port Mismatch
        if (data['src_port'] in [80, 443, 8080] and data['protocol'] == "UDP") or \
                (data['src_port'] == 22 and data['protocol'] == "TCP"):
            return "PROTOCOL_PORT_MISMATCH"

        # Suspicious Off-Hours Activity
        if is_weekend or not is_business_hours:
            if data['src_port'] in common_service_ports and data['packet_size'] > 1500:
                return "OFF_HOURS_SUSPICIOUS_ACTIVITY"

        # High Port with Low Packet Size (potential scanning)
        if (data['src_port'] > 1024 or data['dst_port'] > 1024) and \
                data['packet_size'] < 100 and data['duration_ms'] < 200:
            return "POTENTIAL_PORT_SCAN"

        # Default Behavioral Anomaly
        return "BEHAVIORAL_ANOMALY"

    def determine_context(self, data, anomaly_score, anomaly_type):
        """Determine context for enhanced analysis"""

        current_time = datetime.now()

        context = {
            'severity': self.calculate_severity(anomaly_score),
            'time_of_day': current_time.hour,
            'day_of_week': current_time.weekday(),
            'business_hours': 9 <= current_time.hour <= 17 and current_time.weekday() < 5,
            'weekend': current_time.weekday() >= 5,
            'anomaly_type': anomaly_type,
            'src_port': data['src_port'],
            'packet_size': data['packet_size'],
            'duration_ms': data['duration_ms'],
            'recent_incidents': self.check_recent_incidents(),
            'investigation_mode': False  # Can be set based on user input
        }

        return context

    def calculate_severity(self, anomaly_score):
        """Calculate severity based on anomaly score"""

        if anomaly_score < -0.1:
            return 'critical'
        elif anomaly_score < -0.05:
            return 'high'
        elif anomaly_score < -0.02:
            return 'medium'
        else:
            return 'low'

    def check_recent_incidents(self):
        """Check for recent incidents in the last hour"""

        if not self.analysis_history:
            return False

        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_incidents = [
            incident for incident in self.analysis_history
            if datetime.fromisoformat(incident['timestamp']) > recent_threshold
        ]

        return len(recent_incidents) >= 3

    def anomaly_analysis(self, data, anomaly_score, anomaly_type):
        """Perform enhanced analysis with context awareness and template selection"""
        if not self.llm_analyzer:
            return "⚠️ LLM analyzer not available - API key not configured"

        # Determine context
        context = self.determine_context(data, anomaly_score, anomaly_type)

        # Select optimal template or run multi-template analysis
        if ENABLE_MULTI_TEMPLATE:
            # Multi-template comparison
            logger.info("Running multi-template analysis...")
            comparison = self.llm_analyzer.compare_all_templates(data, anomaly_score, anomaly_type)

            # Save comparison report if enabled
            if SAVE_COMPARISON_REPORTS:
                self.save_comparison_report(comparison, data)

            # Return all template responses and comparison summary
            return {
                'individual_results': comparison['individual_results'],
                'comparison_summary': comparison['comparison_summary']
            }
        else:
            # Single template analysis
            template_name = self.llm_analyzer.get_best_template_for_context(context)
            logger.info(f"Using template: {template_name}")
            result = self.llm_analyzer.analyze_with_template(data, template_name, anomaly_score, anomaly_type)
            return {'single_response': result['llm_response'], 'template_used': template_name}

    def save_comparison_report(self, comparison, data):
        """Save comparison report to file"""

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{COMPARISON_REPORT_DIR}comparison_report_{timestamp}.json"

            report_data = {
                'comparison': comparison,
                'input_data': data,
                'timestamp': datetime.now().isoformat()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Comparison report saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to save comparison report: {str(e)}")


# =============================== Logging and Display ============================== #

def log_anomaly(data, anomaly_score, llm_result, anomaly_type, template_used=None):
    """Enhanced anomaly logging with support for all template responses"""
    try:
        file_exists = os.path.isfile(LOG_FILE)

        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp', 'src_port', 'dst_port', 'packet_size',
                'duration_ms', 'protocol', 'anomaly_score', 'anomaly_type',
                'template_used', 'severity',
                'llm_analysis_technical_expert', 'llm_analysis_risk_assessor',
                'llm_analysis_incident_responder', 'llm_analysis_threat_intel',
                'llm_analysis_executive_briefing'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
                logger.info(f"Created new anomaly log: {LOG_FILE}")

            # Calculate severity
            severity = 'critical' if anomaly_score < -0.1 else 'high' if anomaly_score < -0.05 else 'medium' if anomaly_score < -0.02 else 'low'

            # Initialize row with common fields
            row = {
                'timestamp': datetime.now().isoformat(),
                'src_port': data['src_port'],
                'dst_port': data['dst_port'],
                'packet_size': data['packet_size'],
                'duration_ms': data['duration_ms'],
                'protocol': data['protocol'],
                'anomaly_score': round(anomaly_score, 4),
                'anomaly_type': anomaly_type,
                'template_used': template_used or DEFAULT_TEMPLATE,
                'severity': severity,
                'llm_analysis_technical_expert': '',
                'llm_analysis_risk_assessor': '',
                'llm_analysis_incident_responder': '',
                'llm_analysis_threat_intel': '',
                'llm_analysis_executive_briefing': ''
            }

            # Populate LLM responses
            if isinstance(llm_result, dict) and 'individual_results' in llm_result:
                # Multi-template mode: Log all template responses
                for template_name in ['technical_expert', 'risk_assessor', 'incident_responder', 'threat_intel', 'executive_briefing']:
                    result = llm_result['individual_results'].get(template_name, {})
                    response = result.get('llm_response', '⚠️ Analysis failed') if 'error' not in result else f"⚠️ Error: {result['error']}"
                    # Truncate response to 500 characters, removing newlines
                    response = response.replace('\n', ' ').replace('\r', ' ')[:500]
                    if len(response) == 500:
                        response += '...'
                    row[f'llm_analysis_{template_name}'] = response
            else:
                # Single-template mode: Log only the specified template's response
                response = llm_result.get('single_response', '⚠️ No response').replace('\n', ' ').replace('\r', ' ')[:500]
                if len(response) == 500:
                    response += '...'
                row[f'llm_analysis_{template_used or DEFAULT_TEMPLATE}'] = response

            writer.writerow(row)
            logger.info(f"Anomaly logged to {LOG_FILE}")

    except Exception as e:
        logger.error(f"❌ Error logging anomaly: {e}")


def display_anomaly_alert(data, anomaly_score, llm_response, anomaly_type, template_used=None):
    """Display enhanced anomaly alert with multi-template support and comparison summary"""
    label = anomaly_type
    severity = "CRITICAL" if anomaly_score < -0.1 else "HIGH" if anomaly_score < -0.05 else "MEDIUM" if anomaly_score < -0.02 else "LOW"
    severity_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}

    # Base anomaly alert
    print("\n🚨 Anomaly Detected!")
    print(f"Label: {label}")
    print(f"Reason: Suspicious network activity detected (see detailed analysis below)\n")
    print("🚨" * 25)
    print("🔥 NETWORK SECURITY ANOMALY DETECTED 🔥")
    print("🚨" * 25)
    print(f"{severity_colors.get(severity, '⚪')} SEVERITY: {severity}")

    print(f"\n📊 TRAFFIC ANALYSIS:")
    print(f"   🔌 Source Port: {data['src_port']}")
    print(f"   🎯 Destination Port: {data['dst_port']}")
    print(f"   📦 Packet Size: {data['packet_size']:,} bytes")
    print(f"   ⏱️ Duration: {data['duration_ms']} ms")
    print(f"   🔗 Protocol: {data['protocol']}")

    print(f"\n🎯 DETECTION METRICS:")
    print(f"   📈 Anomaly Score: {anomaly_score:.4f}")
    print(f"   🏷️ Classification: {anomaly_type}")
    print(f"   📅 Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Display LLM analysis
    print(f"\n🧠 AI SECURITY ANALYSIS:")
    print("   " + "─" * 50)

    if isinstance(llm_response, dict) and 'individual_results' in llm_response:
        # Multi-template mode: Display all template responses
        print("   🔄 Multi-Template Analysis Results:")
        for template_name, result in llm_response['individual_results'].items():
            print(f"\n   📋 Template: {template_name.title()}")
            print(f"   📝 Description: {result['template_description']}")
            if 'error' in result:
                print(f"   ⚠️ Error: {result['error']}")
            else:
                formatted_response = result['llm_response'].replace('\n', '\n   ')
                print(f"   {formatted_response}")
            print("   " + "─" * 50)

        # Display comparison summary
        print(f"\n   📊 Comparison Summary:")
        comparison = llm_response['comparison_summary']
        print(f"   • Successful Analyses: {comparison['success_count']}")
        print(f"   • Failed Analyses: {comparison['error_count']}")
        print(f"\n   Response Times:")
        for template_name, time in comparison['response_times'].items():
            print(f"     • {template_name.title()}: {time:.2f}s")
        print(f"\n   Response Lengths (words):")
        for template_name, length in comparison['response_lengths'].items():
            print(f"     • {template_name.title()}: {length} words")
        print(f"\n   Severity Assessments:")
        for template_name, severity in comparison['severity_assessments'].items():
            print(f"     • {template_name.title()}: {severity}")
        print("   " + "─" * 50)
    else:
        # Single-template mode
        print(f"   🤖 Analysis Template: {template_used or DEFAULT_TEMPLATE}")
        formatted_response = llm_response['single_response'].replace('\n', '\n   ')
        print(f"   {formatted_response}")
        print("   " + "─" * 50)

    print("\n" + "🚨" * 25)


def print_startup_info():
    """Print enhanced startup information and configuration"""

    print("🛡️" + "=" * 70)
    print("🔍 NETWORK ANOMALY DETECTION SYSTEM")
    print("🛡️" + "=" * 70)

    print(f"📡 Server Configuration:")
    print(f"   • Host: {HOST}:{PORT}")
    print(f"   • Log File: {LOG_FILE}")
    print(f"   • Performance Log: {PERFORMANCE_LOG}")

    print(f"\n🤖 AI Configuration:")
    print(f"   • Model: {MODEL_NAME}")
    print(f"   • Max Tokens: {MAX_TOKENS}")
    print(f"   • Temperature: {TEMPERATURE}")
    print(f"   • Default Template: {DEFAULT_TEMPLATE}")

    print(f"\n🔧 Enhanced Features:")
    print(f"   • Multi-Template Analysis: {'✅ Enabled' if ENABLE_MULTI_TEMPLATE else '❌ Disabled'}")
    print(f"   • Report Directory: {COMPARISON_REPORT_DIR}")
    api_status = "✅ Configured" if TOGETHER_API_KEY and TOGETHER_API_KEY != "your_together_ai_api_key_here" else\
        "❌ Not configured"
    print(f"   • API Status: {api_status}")

    print("🛡️" + "=" * 70)


def display_performance_summary(detector):
    """Display performance summary of templates"""

    if not detector.llm_analyzer or not detector.llm_analyzer.performance_metrics:
        return

    print("\n📊 TEMPLATE PERFORMANCE SUMMARY:")
    print("─" * 50)

    for template_name, metrics in detector.llm_analyzer.performance_metrics.items():
        if metrics['usage_count'] > 0:
            avg_time = statistics.mean(metrics['response_times']) if metrics['response_times'] else 0
            avg_length = statistics.mean(metrics['response_lengths']) if metrics['response_lengths'] else 0

            print(f"🔹 {template_name.title()}: {metrics['usage_count']} uses, "
                  f"avg {avg_time:.2f}s, {avg_length:.0f} words")


# =============================== Main Client ======================================= #
def main():
    """main client loop with context-aware anomaly detection and multi-template support"""
    print_startup_info()

    # Initialize enhanced detector
    detector = ContextAwareAnomalyDetector()

    print("📡 Connecting to data stream...")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            buffer = ""
            logger.info("Client connected to server")

            packet_count = 0
            anomaly_count = 0
            last_performance_save = time.time()

            while True:
                try:
                    chunk = s.recv(1024).decode()
                    if not chunk:
                        logger.info("Server disconnected")
                        break

                    buffer += chunk

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        try:
                            data = json.loads(line)
                            packet_count += 1

                            print(f'📥 Packet #{packet_count}: {data}')
                            if packet_count == os.getenv("PACKET_COUNT", 100):
                                raise KeyboardInterrupt

                            # Detect anomaly
                            is_anomaly, anomaly_score = detector.detect_anomaly(data)

                            if is_anomaly:
                                anomaly_count += 1
                                anomaly_type = detector.classify_anomaly_type(data)
                                logger.info(f"Anomaly detected: {anomaly_type} (score: {anomaly_score:.4f})")

                                # Enhanced LLM analysis
                                template_used = None
                                try:
                                    llm_result = detector.anomaly_analysis(data, anomaly_score, anomaly_type)

                                    # Handle multi-template vs single-template results
                                    if isinstance(llm_result, dict) and 'individual_results' in llm_result:
                                        llm_response = llm_result  # Pass the entire result for multi-template
                                        template_used = "multi_template_comparison"
                                    else:
                                        llm_response = llm_result
                                        if detector.llm_analyzer:
                                            context = detector.determine_context(data, anomaly_score, anomaly_type)
                                            template_used = llm_result.get('template_used', detector.llm_analyzer.get_best_template_for_context(context))

                                    # Store analysis result
                                    detector.analysis_history.append({
                                        'timestamp': datetime.now().isoformat(),
                                        'data': data,
                                        'anomaly_score': anomaly_score,
                                        'anomaly_type': anomaly_type,
                                        'template_used': template_used,
                                        'severity': detector.calculate_severity(anomaly_score)
                                    })

                                    # Keep only recent history (last 100)
                                    if len(detector.analysis_history) > 100:
                                        detector.analysis_history = detector.analysis_history[-100:]

                                except Exception as e:
                                    logger.error(f"LLM analysis failed: {str(e)}")
                                    llm_response = {'single_response': f"⚠️ LLM analysis failed: {str(e)}"}
                                    template_used = template_used or DEFAULT_TEMPLATE

                                # Display alert with all template responses and comparison summary (if multi-template)
                                display_anomaly_alert(data, anomaly_score, llm_response, anomaly_type, template_used)

                                # Log the anomaly with all template responses
                                log_anomaly(data, anomaly_score, llm_response, anomaly_type, template_used)

                                # Display statistics
                                anomaly_rate = (anomaly_count / packet_count) * 100
                                print(f"📊 Statistics: {anomaly_count} anomalies detected out of {packet_count} packets ({anomaly_rate:.1f}%)")

                                # Display performance summary every 10 anomalies
                                if anomaly_count % 10 == 0:
                                    display_performance_summary(detector)

                            else:
                                print(f"✅ Normal traffic (score: {anomaly_score:.3f})")

                            print()  # Add spacing between entries

                            # Save performance metrics periodically (every 5 minutes)
                            current_time = time.time()
                            if current_time - last_performance_save > 300:  # 5 minutes
                                if detector.llm_analyzer:
                                    detector.llm_analyzer.save_performance_metrics()
                                last_performance_save = current_time

                        except json.JSONDecodeError:
                            logger.error("Error decoding JSON data")
                        except Exception as e:
                            logger.error(f"Error processing packet: {str(e)}")

                except KeyboardInterrupt:
                    print(f"\n🛑 Shutting down gracefully...")

                    # Save final performance metrics
                    if detector.llm_analyzer:
                        detector.llm_analyzer.save_performance_metrics()

                    # Display final statistics
                    print(f"\n📊 FINAL STATISTICS:")
                    print(f"   📦 Total packets processed: {packet_count:,}")
                    print(f"   🚨 Anomalies detected: {anomaly_count:,}")

                    if packet_count > 0:
                        anomaly_rate = (anomaly_count / packet_count) * 100
                        print(f"   📈 Anomaly rate: {anomaly_rate:.2f}%")

                    # Display template performance summary
                    display_performance_summary(detector)

                    # Analyze recent trends
                    if detector.analysis_history:
                        recent_critical = sum(1 for a in detector.analysis_history if a['severity'] == 'critical')
                        recent_high = sum(1 for a in detector.analysis_history if a['severity'] == 'high')

                        print(f"\n🎯 RECENT THREAT ANALYSIS:")
                        print(f"   🔴 Critical threats: {recent_critical}")
                        print(f"   🟠 High-severity threats: {recent_high}")

                        # Most common anomaly types
                        anomaly_types = [a['anomaly_type'] for a in detector.analysis_history]
                        if anomaly_types:
                            from collections import Counter
                            most_common = Counter(anomaly_types).most_common(3)
                            print(f"   🏷️ Most common threats: {', '.join([f'{t}({c})' for t, c in most_common])}")

                    print(f"\n💾 Data saved to:")
                    print(f"   📝 Anomaly log: {LOG_FILE}")
                    print(f"   📊 Performance metrics: {PERFORMANCE_LOG}")
                    if SAVE_COMPARISON_REPORTS:
                        print(f"   📋 Comparison reports: {COMPARISON_REPORT_DIR}")

                    print("👋 System shutdown complete!")
                    break

                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    continue

    except ConnectionRefusedError:
        logger.error("Connection refused. Make sure the server is running (python server.py)")
        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("   1. Start the server: python server.py")
        print("   2. Check if port 9999 is available")
        print("   3. Verify HOST and PORT configuration")

    except Exception as e:
        logger.error(f"Connection error: {e}")
        print(f"\n❌ Failed to connect: {str(e)}")


if __name__ == "__main__":
    import argparse
    import sys
    import logging

    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="Network Anomaly Detection Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["technical_expert", "risk_assessor", "incident_responder", "threat_intel", "executive_briefing"],
        help="Override default template for LLM analysis (disables multi-template if specified)"
    )
    parser.add_argument(
        "--no-multi-template",
        action="store_true",
        help="Disable multi-template analysis (enabled by default)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=HOST,
        help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help="Server port number"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Enable saving of comparison reports",
        default=SAVE_COMPARISON_REPORTS
    )

    args = parser.parse_args()

    # Set multi-template as default
    ENABLE_MULTI_TEMPLATE = not args.no_multi_template  # Enable by default, disable with --no-multi-template

    # Update configuration based on arguments
    if args.template:
        DEFAULT_TEMPLATE = args.template
        ENABLE_MULTI_TEMPLATE = False  # Disable multi-template if specific template is chosen
        logging.info(f"Using single template: {DEFAULT_TEMPLATE}")
    else:
        logging.info(f"Multi-template analysis: {'Enabled' if ENABLE_MULTI_TEMPLATE else 'Disabled'}")

    if args.host:
        HOST = args.host
        logging.info(f"Server host set to: {HOST}")

    if args.port:
        PORT = args.port
        logging.info(f"Server port set to: {PORT}")

    if args.save_reports:
        SAVE_COMPARISON_REPORTS = True
        logging.info("Comparison report saving enabled")

    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logging.info(f"Logging level set to: {args.log_level}")

    # Validate configuration
    try:
        if not TOGETHER_API_KEY or TOGETHER_API_KEY == "your_together_ai_api_key_here":
            logging.error("Together AI API key not configured in .env file")
            sys.exit(1)

        if args.template and args.template not in [
            "technical_expert", "risk_assessor", "incident_responder",
            "threat_intel", "executive_briefing"
        ]:
            logging.error(f"Invalid template: {args.template}")
            sys.exit(1)

        # Display startup configuration
        print("🛡️ Starting Network Anomaly Detection Client")
        print(f"   🔧 Template: {DEFAULT_TEMPLATE if not ENABLE_MULTI_TEMPLATE else 'All Templates (Multi-Template)'}")
        print(f"   🔄 Multi-template: {'Enabled' if ENABLE_MULTI_TEMPLATE else 'Disabled'}")
        print(f"   📡 Server: {HOST}:{PORT}")
        print(f"   📝 Save Reports: {'Enabled' if SAVE_COMPARISON_REPORTS else 'Disabled'}")
        print("─" * 50)

        main()

    except KeyboardInterrupt:
        logging.info("Program terminated by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Startup failed: {str(e)}")
        sys.exit(1)