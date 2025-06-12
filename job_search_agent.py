import streamlit as st
import json
import time
from datetime import datetime
import re
from typing import List, Dict, Any
import pandas as pd

# Try to import Exa, fall back to mock if not available
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    st.error("""
    **⚠️ Exa package not found!**
    
    To use real web crawling, install the exa package:
    ```
    pip install exa_py
    ```
    
    For now, the app will run in **demo mode** with mock data.
    """)

# Initialize Exa client
@st.cache_resource
def init_exa():
    if EXA_AVAILABLE:
        return Exa(api_key="ee6f2bad-90d7-442c-a892-6f5cf2f611ab")
    else:
        return None

# Initialize session state for memory
if 'memory' not in st.session_state:
    st.session_state.memory = {
        'previous_searches': [],
        'preferences': {},
        'saved_jobs': [],
        'search_history': []
    }

if 'current_step' not in st.session_state:
    st.session_state.current_step = 'idle'

if 'planning_results' not in st.session_state:
    st.session_state.planning_results = None

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

if 'summary' not in st.session_state:
    st.session_state.summary = ""

class JobSearchAgent:
    def __init__(self, exa_client):
        self.exa = exa_client
        self.job_sites = [
            "linkedin.com/jobs",
            "indeed.com",
            "glassdoor.com",
            "stackoverflow.com/jobs",
            "angel.co",
            "wellfound.com",
            "remote.co",
            "weworkremotely.com",
            "remoteok.io",
            "jobs.ashbyhq.com"
        ]
    
    def analyze_intent(self, query: str) -> str:
        """Analyze user intent from search query"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['remote', 'work from home', 'distributed']):
            return 'remote_work'
        elif any(keyword in query_lower for keyword in ['senior', 'lead', 'principal', 'staff']):
            return 'senior_roles'
        elif any(keyword in query_lower for keyword in ['junior', 'entry', 'graduate', 'intern']):
            return 'entry_level'
        elif any(keyword in query_lower for keyword in ['ai', 'machine learning', 'ml', 'data scientist']):
            return 'ai_ml'
        elif any(keyword in query_lower for keyword in ['frontend', 'react', 'vue', 'angular']):
            return 'frontend'
        elif any(keyword in query_lower for keyword in ['backend', 'api', 'server', 'database']):
            return 'backend'
        elif any(keyword in query_lower for keyword in ['fullstack', 'full stack', 'full-stack']):
            return 'fullstack'
        else:
            return 'general_search'
    
    def create_search_strategy(self, intent: str) -> str:
        """Create search strategy based on detected intent"""
        strategies = {
            'remote_work': 'Focus on remote-first companies and distributed teams',
            'senior_roles': 'Target leadership positions and high-impact roles',
            'entry_level': 'Find companies with strong mentorship and growth programs',
            'ai_ml': 'Search AI/ML focused companies and data-driven roles',
            'frontend': 'Look for UI/UX focused roles and modern frontend technologies',
            'backend': 'Search for API, database, and server-side positions',
            'fullstack': 'Find roles requiring both frontend and backend skills',
            'general_search': 'Broad search across multiple criteria and platforms'
        }
        return strategies.get(intent, strategies['general_search'])
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from search query"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords
    
    def plan_search(self, query: str, filters: Dict) -> Dict:
        """Step 1: Planning Phase"""
        intent = self.analyze_intent(query)
        strategy = self.create_search_strategy(intent)
        keywords = self.extract_keywords(query)
        
        # Estimate results based on query complexity
        estimated_results = min(50, max(10, len(keywords) * 8))
        
        return {
            'intent': intent,
            'strategy': strategy,
            'keywords': keywords,
            'estimated_results': estimated_results,
            'search_terms': self.build_search_terms(query, intent, keywords)
        }
    
    def build_search_terms(self, query: str, intent: str, keywords: List[str]) -> List[str]:
        """Build optimized search terms for Exa"""
        base_terms = []
        
        # Add the original query
        base_terms.append(f"jobs {query}")
        
        # Add intent-specific terms
        if intent == 'remote_work':
            base_terms.extend([
                f"remote {' '.join(keywords[:3])} jobs",
                f"work from home {' '.join(keywords[:2])}"
            ])
        elif intent == 'ai_ml':
            base_terms.extend([
                f"artificial intelligence {' '.join(keywords[:2])} jobs",
                f"machine learning engineer positions"
            ])
        elif intent == 'senior_roles':
            base_terms.extend([
                f"senior {' '.join(keywords[:2])} positions",
                f"lead {' '.join(keywords[:2])} jobs"
            ])
        
        # Add general job search terms
        base_terms.extend([
            f"hiring {' '.join(keywords[:3])}",
            f"career opportunities {' '.join(keywords[:2])}"
        ])
        
        return base_terms[:5]  # Limit to 5 search terms
    
    def execute_search(self, plan: Dict, filters: Dict) -> List[Dict]:
        """Step 2: Search Phase using Exa AI or Mock Data"""
        if self.exa and EXA_AVAILABLE:
            return self._execute_real_search(plan, filters)
        else:
            return self._execute_mock_search(plan, filters)
    
    def _execute_real_search(self, plan: Dict, filters: Dict) -> List[Dict]:
        """Real search using Exa AI"""
        all_results = []
        
        for search_term in plan['search_terms']:
            try:
                # Use Exa to search for job-related content
                result = self.exa.search_and_contents(
                    query=search_term,
                    num_results=10,
                    text=True,
                    include_domains=self.job_sites,
                    use_autoprompt=True
                )
                
                # Process and extract job information
                for item in result.results:
                    job_data = self.extract_job_info(item, plan['keywords'])
                    if job_data and self.apply_logic_gates(job_data, filters):
                        all_results.append(job_data)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                st.warning(f"Search error for '{search_term}': {str(e)}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_jobs = self.deduplicate_jobs(all_results)
        return sorted(unique_jobs, key=lambda x: x.get('relevance_score', 0), reverse=True)[:20]
    
    def _execute_mock_search(self, plan: Dict, filters: Dict) -> List[Dict]:
        """Mock search for demonstration when Exa is not available"""
        # Simulate search delay
        time.sleep(2)
        
        # Mock job data based on search intent
        mock_jobs = self._generate_mock_jobs(plan)
        
        # Apply filters
        filtered_jobs = [job for job in mock_jobs if self.apply_logic_gates(job, filters)]
        
        return filtered_jobs[:20]
    
    def extract_job_info(self, exa_result, keywords: List[str]) -> Dict:
        """Extract job information from Exa search result"""
        text = exa_result.text.lower() if exa_result.text else ""
        title = exa_result.title or "Job Position"
        url = exa_result.url
        
        # Extract job details using regex and keyword matching
        job_data = {
            'id': hash(url),
            'title': self.extract_job_title(title, text),
            'company': self.extract_company_name(url, text),
            'location': self.extract_location(text),
            'salary': self.extract_salary(text),
            'job_type': self.extract_job_type(text),
            'experience': self.extract_experience_level(text),
            'description': text[:300] + "..." if len(text) > 300 else text,
            'url': url,
            'tags': self.extract_tags(text, keywords),
            'relevance_score': self.calculate_relevance(text, keywords),
            'source': self.get_source_site(url)
        }
        
        return job_data
    
    def extract_job_title(self, title: str, text: str) -> str:
        """Extract job title from content"""
        # Common job title patterns
        title_patterns = [
            r'(?:position|role|job).*?([a-zA-Z\s]+(?:engineer|developer|scientist|manager|analyst|designer))',
            r'(senior|junior|lead|principal|staff)\s+([a-zA-Z\s]+(?:engineer|developer|scientist))',
            r'hiring.*?([a-zA-Z\s]+(?:engineer|developer|scientist|manager))'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return title.split(' - ')[0] if ' - ' in title else title
    
    def extract_company_name(self, url: str, text: str) -> str:
        """Extract company name from URL or text"""
        # Extract from URL domain
        domain_match = re.search(r'https?://(?:www\.)?([^./]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            if domain not in ['linkedin', 'indeed', 'glassdoor', 'stackoverflow']:
                return domain.replace('-', ' ').title()
        
        # Try to extract from text
        company_patterns = [
            r'(?:company|at|join)\s+([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Ltd)?)',
            r'([A-Z][a-zA-Z\s&]+)\s+is\s+(?:hiring|looking|seeking)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return "Company Name"
    
    def extract_location(self, text: str) -> str:
        """Extract location from job text"""
        location_patterns = [
            r'(?:location|based in|located in)\s*:?\s*([A-Z][a-zA-Z\s,]+(?:CA|NY|TX|FL|WA|Remote))',
            r'(Remote|Work from home)',
            r'([A-Z][a-zA-Z\s]+,\s*[A-Z]{2})',  # City, State
            r'([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)'  # City, Country
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"
    
    def extract_salary(self, text: str) -> str:
        """Extract salary information from text"""
        salary_patterns = [
            r'\$(\d{2,3}[,.]?\d{3})\s*-?\s*\$?(\d{2,3}[,.]?\d{3})?\s*(?:k|thousand|per year|annually)?',
            r'(\d{2,3})\s*-\s*(\d{2,3})\s*k',
            r'salary.*?\$(\d{2,3}[,.]?\d{3})'
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"${match.group(1)}{f' - ${match.group(2)}' if match.group(2) else ''}"
        
        return "Not specified"
    
    def extract_job_type(self, text: str) -> str:
        """Extract job type (full-time, contract, etc.)"""
        if re.search(r'full.?time', text, re.IGNORECASE):
            return 'Full-time'
        elif re.search(r'part.?time', text, re.IGNORECASE):
            return 'Part-time'
        elif re.search(r'contract|contractor', text, re.IGNORECASE):
            return 'Contract'
        elif re.search(r'intern|internship', text, re.IGNORECASE):
            return 'Internship'
        else:
            return 'Full-time'  # Default assumption
    
    def extract_experience_level(self, text: str) -> str:
        """Extract experience level from text"""
        if re.search(r'senior|lead|principal|staff', text, re.IGNORECASE):
            return 'Senior'
        elif re.search(r'junior|entry|graduate|new grad', text, re.IGNORECASE):
            return 'Junior'
        elif re.search(r'mid.?level|intermediate', text, re.IGNORECASE):
            return 'Mid-level'
        else:
            return 'Mid-level'  # Default assumption
    
    def extract_tags(self, text: str, keywords: List[str]) -> List[str]:
        """Extract relevant tags/skills from job text"""
        # Common technology and skill keywords
        tech_keywords = [
            'python', 'javascript', 'java', 'react', 'angular', 'vue', 'node',
            'django', 'flask', 'spring', 'sql', 'mongodb', 'postgresql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
            'machine learning', 'ai', 'data science', 'tensorflow', 'pytorch'
        ]
        
        found_tags = []
        text_lower = text.lower()
        
        # Add matching keywords from search
        for keyword in keywords:
            if keyword in text_lower:
                found_tags.append(keyword.title())
        
        # Add matching tech keywords
        for tech in tech_keywords:
            if tech in text_lower:
                found_tags.append(tech.title())
        
        return list(set(found_tags))[:8]  # Limit to 8 tags
    
    def calculate_relevance(self, text: str, keywords: List[str]) -> int:
        """Calculate relevance score based on keyword matches"""
        text_lower = text.lower()
        score = 50  # Base score
        
        # Add points for keyword matches
        for keyword in keywords:
            if keyword in text_lower:
                score += 10
        
        # Add points for job-related terms
        job_terms = ['hiring', 'position', 'role', 'opportunity', 'career']
        for term in job_terms:
            if term in text_lower:
                score += 5
        
        return min(100, score)
    
    def get_source_site(self, url: str) -> str:
        """Get the source website name"""
        if 'linkedin' in url:
            return 'LinkedIn'
        elif 'indeed' in url:
            return 'Indeed'
        elif 'glassdoor' in url:
            return 'Glassdoor'
        elif 'stackoverflow' in url:
            return 'Stack Overflow'
        elif 'angel.co' in url or 'wellfound' in url:
            return 'Wellfound'
        else:
            domain = re.search(r'https?://(?:www\.)?([^./]+)', url)
            return domain.group(1).title() if domain else 'Unknown'
    
    def deduplicate_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs based on title and company"""
        seen = set()
        unique_jobs = []
        
        for job in jobs:
            key = (job['title'].lower(), job['company'].lower())
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        return unique_jobs
    
    def apply_logic_gates(self, job: Dict, filters: Dict) -> bool:
        """Apply logic gates to filter jobs based on criteria"""
        # Salary filter logic gate
        if filters.get('min_salary'):
            job_salary_text = job.get('salary', '').replace('$', '').replace(',', '')
            salary_match = re.search(r'(\d+)', job_salary_text)
            if salary_match:
                job_salary = int(salary_match.group(1))
                # Convert to full number if it looks like abbreviated (e.g., "120" -> 120000)
                if job_salary < 1000:
                    job_salary *= 1000
                if job_salary < int(filters['min_salary']):
                    return False
        
        # Location filter logic gate
        if filters.get('location') and filters['location'] != 'Any':
            job_location = job.get('location', '').lower()
            filter_location = filters['location'].lower()
            if filter_location not in job_location and 'remote' not in job_location:
                return False
        
        # Job type filter logic gate
        if filters.get('job_type') and filters['job_type'] != 'Any':
            if job.get('job_type', '').lower() != filters['job_type'].lower():
                return False
        
        # Experience filter logic gate
        if filters.get('experience') and filters['experience'] != 'Any':
            if job.get('experience', '').lower() != filters['experience'].lower():
                return False
        
        return True
    
    def generate_summary(self, results: List[Dict], plan: Dict) -> str:
        """Step 3: Generate summary of search results"""
        if not results:
            return "No jobs found matching your criteria. Try adjusting your search terms or filters."
        
        total_jobs = len(results)
        
        # Calculate salary statistics
        salaries = []
        for job in results:
            salary_text = job.get('salary', '').replace('$', '').replace(',', '')
            salary_match = re.search(r'(\d+)', salary_text)
            if salary_match:
                salary = int(salary_match.group(1))
                if salary < 1000:  # Convert abbreviated salaries
                    salary *= 1000
                salaries.append(salary)
        
        avg_salary = sum(salaries) / len(salaries) if salaries else 0
        
        # Count remote jobs
        remote_jobs = sum(1 for job in results if 'remote' in job.get('location', '').lower())
        
        # Get top companies
        companies = [job['company'] for job in results[:5]]
        top_companies = list(dict.fromkeys(companies))[:3]  # Remove duplicates, keep order
        
        # Get source distribution
        sources = {}
        for job in results:
            source = job.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        summary_parts = [
            f"Found {total_jobs} job opportunities matching your search for '{plan.get('keywords', ['your query'])[0] if plan.get('keywords') else 'jobs'}'."
        ]
        
        if avg_salary > 0:
            summary_parts.append(f"Average salary: ${int(avg_salary):,}")
        
        if remote_jobs > 0:
            summary_parts.append(f"{remote_jobs} remote positions available")
        
        if top_companies:
            summary_parts.append(f"Top companies: {', '.join(top_companies)}")
        
        if sources:
            top_source = max(sources.items(), key=lambda x: x[1])
            summary_parts.append(f"Most results from: {top_source[0]} ({top_source[1]} jobs)")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_mock_jobs(self, plan: Dict) -> List[Dict]:
        """Generate mock jobs based on search intent for demo purposes"""
        intent = plan.get('intent', 'general_search')
        keywords = plan.get('keywords', [])
        
        # Base mock jobs that adapt to search intent
        base_jobs = [
            {
                'id': 1,
                'title': 'Senior Software Engineer',
                'company': 'TechCorp AI',
                'location': 'Remote',
                'salary': '$120,000 - $150,000',
                'job_type': 'Full-time',
                'experience': 'Senior',
                'description': 'Join our AI team building next-generation applications with Python and machine learning.',
                'url': 'https://techcorp.ai/careers/senior-engineer',
                'tags': ['Python', 'AI/ML', 'Remote', 'Senior'],
                'relevance_score': 95,
                'source': 'Company Website'
            },
            {
                'id': 2,
                'title': 'Frontend Developer',
                'company': 'WebSolutions Inc',
                'location': 'New York, NY',
                'salary': '$80,000 - $100,000',
                'job_type': 'Full-time',
                'experience': 'Mid-level',
                'description': 'Create beautiful user interfaces with React and TypeScript for modern web applications.',
                'url': 'https://websolutions.com/jobs/frontend-dev',
                'tags': ['React', 'TypeScript', 'Frontend', 'Mid-level'],
                'relevance_score': 87,
                'source': 'LinkedIn'
            },
            {
                'id': 3,
                'title': 'Data Scientist',
                'company': 'DataDriven Co',
                'location': 'San Francisco, CA',
                'salary': '$110,000 - $140,000',
                'job_type': 'Full-time',
                'experience': 'Senior',
                'description': 'Analyze complex datasets and build predictive models using Python and machine learning.',
                'url': 'https://datadriven.co/careers/data-scientist',
                'tags': ['Python', 'Data Science', 'ML', 'Senior'],
                'relevance_score': 92,
                'source': 'Indeed'
            },
            {
                'id': 4,
                'title': 'Junior Web Developer',
                'company': 'StartupXYZ',
                'location': 'Austin, TX',
                'salary': '$55,000 - $70,000',
                'job_type': 'Full-time',
                'experience': 'Junior',
                'description': 'Great opportunity for new graduates to learn and grow with modern web technologies.',
                'url': 'https://startupxyz.com/jobs/junior-dev',
                'tags': ['JavaScript', 'HTML/CSS', 'Junior', 'Growth'],
                'relevance_score': 78,
                'source': 'Glassdoor'
            },
            {
                'id': 5,
                'title': 'DevOps Engineer',
                'company': 'CloudFirst',
                'location': 'Remote',
                'salary': '$95,000 - $125,000',
                'job_type': 'Contract',
                'experience': 'Mid-level',
                'description': 'Manage cloud infrastructure and deployment pipelines using AWS and Kubernetes.',
                'url': 'https://cloudfirst.io/careers/devops',
                'tags': ['AWS', 'Docker', 'Kubernetes', 'Remote'],
                'relevance_score': 89,
                'source': 'Stack Overflow'
            },
            {
                'id': 6,
                'title': 'Machine Learning Engineer',
                'company': 'AI Innovations',
                'location': 'Remote',
                'salary': '$130,000 - $160,000',
                'job_type': 'Full-time',
                'experience': 'Senior',
                'description': 'Build and deploy ML models at scale using TensorFlow and PyTorch.',
                'url': 'https://aiinnovations.com/jobs/ml-engineer',
                'tags': ['Python', 'TensorFlow', 'PyTorch', 'ML'],
                'relevance_score': 96,
                'source': 'Wellfound'
            }
        ]
        
        # Adjust jobs based on search intent
        if intent == 'remote_work':
            # Prioritize remote jobs
            for job in base_jobs:
                if job['location'] == 'Remote':
                    job['relevance_score'] += 10
        elif intent == 'ai_ml':
            # Boost AI/ML related jobs
            for job in base_jobs:
                if any(tag in ['AI/ML', 'ML', 'Data Science', 'TensorFlow', 'PyTorch'] for tag in job['tags']):
                    job['relevance_score'] += 15
        elif intent == 'senior_roles':
            # Boost senior positions
            for job in base_jobs:
                if job['experience'] == 'Senior':
                    job['relevance_score'] += 12
        elif intent == 'entry_level':
            # Boost junior positions
            for job in base_jobs:
                if job['experience'] == 'Junior':
                    job['relevance_score'] += 15
        
        # Adjust relevance based on keyword matches
        for job in base_jobs:
            for keyword in keywords:
                job_text = f"{job['title']} {job['description']} {' '.join(job['tags'])}".lower()
                if keyword.lower() in job_text:
                    job['relevance_score'] = min(100, job['relevance_score'] + 5)
        
        return base_jobs

def main():
    st.set_page_config(
        page_title="AI Job Search Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize Exa client
    exa = init_exa()
    agent = JobSearchAgent(exa)
    
    # Header
    st.title("🤖 AI Job Search Agent")
    if EXA_AVAILABLE:
        st.markdown("**Multi-step workflow: Plan → Search → Summarize with Exa AI Web Crawling**")
    else:
        st.markdown("**Multi-step workflow: Plan → Search → Summarize (Demo Mode)**")
        st.info("💡 Install `exa_py` package to enable real web crawling capabilities!")
    st.markdown("---")
    
    # Sidebar for filters
    with st.sidebar:
        st.header("🔧 Search Filters")
        
        location_filter = st.selectbox(
            "Location",
            ["Any", "Remote", "New York, NY", "San Francisco, CA", "Seattle, WA", "Austin, TX", "Boston, MA"]
        )
        
        min_salary = st.number_input(
            "Minimum Salary ($)",
            min_value=0,
            max_value=500000,
            step=5000,
            value=0
        )
        
        job_type = st.selectbox(
            "Job Type",
            ["Any", "Full-time", "Part-time", "Contract", "Internship"]
        )
        
        experience = st.selectbox(
            "Experience Level",
            ["Any", "Junior", "Mid-level", "Senior"]
        )
        
        filters = {
            'location': location_filter,
            'min_salary': str(min_salary) if min_salary > 0 else '',
            'job_type': job_type,
            'experience': experience
        }
        
        # Memory section
        st.header("🧠 Agent Memory")
        if st.session_state.memory['previous_searches']:
            st.write("**Recent Searches:**")
            for search in st.session_state.memory['previous_searches'][-3:]:
                st.write(f"• \"{search['query']}\" - {search['result_count']} results")
        else:
            st.write("No previous searches")
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "What kind of job are you looking for?",
            placeholder="e.g., Senior Python developer at AI company, Remote frontend jobs, Data scientist positions...",
            value=""
        )
    
    with col2:
        search_button = st.button("🔍 Search Jobs", type="primary", disabled=not query.strip())
    
    # Workflow status
    if st.session_state.current_step != 'idle':
        st.header("🔄 Agent Workflow Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.current_step == 'planning':
                st.info("🧠 **Planning** (Current)")
            elif st.session_state.current_step in ['searching', 'summarizing', 'complete']:
                st.success("🧠 **Planning** (Complete)")
            else:
                st.write("🧠 **Planning**")
        
        with col2:
            if st.session_state.current_step == 'searching':
                st.info("🔍 **Searching** (Current)")
            elif st.session_state.current_step in ['summarizing', 'complete']:
                st.success("🔍 **Searching** (Complete)")
            else:
                st.write("🔍 **Searching**")
        
        with col3:
            if st.session_state.current_step == 'summarizing':
                st.info("📊 **Summarizing** (Current)")
            elif st.session_state.current_step == 'complete':
                st.success("📊 **Summarizing** (Complete)")
            else:
                st.write("📊 **Summarizing**")
    
    # Execute search
    if search_button and query.strip():
        try:
            # Step 1: Planning Phase
            st.session_state.current_step = 'planning'
            st.rerun()
            
            with st.spinner("🧠 Planning your job search..."):
                plan = agent.plan_search(query, filters)
                st.session_state.planning_results = plan
                time.sleep(1)  # Brief pause for UX
            
            # Step 2: Search Phase
            st.session_state.current_step = 'searching'
            st.rerun()
            
            with st.spinner("🔍 Searching job sites with Exa AI..."):
                results = agent.execute_search(plan, filters)
                st.session_state.search_results = results
            
            # Step 3: Summarization Phase
            st.session_state.current_step = 'summarizing'
            st.rerun()
            
            with st.spinner("📊 Analyzing and summarizing results..."):
                summary = agent.generate_summary(results, plan)
                st.session_state.summary = summary
                time.sleep(1)  # Brief pause for UX
            
            # Update memory
            st.session_state.memory['previous_searches'].append({
                'query': query,
                'filters': filters,
                'result_count': len(results),
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 searches
            st.session_state.memory['previous_searches'] = st.session_state.memory['previous_searches'][-10:]
            
            st.session_state.current_step = 'complete'
            st.rerun()
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            st.session_state.current_step = 'idle'
    
    # Display planning results
    if st.session_state.planning_results:
        st.header("🧠 Planning Phase Results")
        plan = st.session_state.planning_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Detected Intent:** {plan['intent'].replace('_', ' ').title()}")
            st.info(f"**Keywords:** {', '.join(plan['keywords'])}")
        
        with col2:
            st.info(f"**Search Strategy:** {plan['strategy']}")
            st.info(f"**Expected Results:** ~{plan['estimated_results']} jobs")
    
    # Display summary
    if st.session_state.summary:
        st.header("📊 Search Summary")
        st.success(st.session_state.summary)
    
    # Display results
    if st.session_state.search_results:
        st.header(f"🎯 Job Matches ({len(st.session_state.search_results)} found)")
        
        for i, job in enumerate(st.session_state.search_results):
            with st.expander(f"**{job['title']}** at **{job['company']}** ({job['relevance_score']}% match)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"📍 **Location:** {job['location']}")
                    st.write(f"💰 **Salary:** {job['salary']}")
                
                with col2:
                    st.write(f"⏰ **Type:** {job['job_type']}")
                    st.write(f"📈 **Level:** {job['experience']}")
                
                with col3:
                    st.write(f"🌐 **Source:** {job['source']}")
                    if st.button(f"⭐ Save Job", key=f"save_{job['id']}"):
                        if job not in st.session_state.memory['saved_jobs']:
                            st.session_state.memory['saved_jobs'].append(job)
                            st.success("Job saved!")
                
                st.write(f"**Description:** {job['description']}")
                
                if job['tags']:
                    st.write("**Tags:** " + " • ".join([f"`{tag}`" for tag in job['tags']]))
                
                st.write(f"**[View Original Job Posting]({job['url']})**")
    
    # Saved jobs section
    if st.session_state.memory['saved_jobs']:
        st.header("⭐ Saved Jobs")
        for job in st.session_state.memory['saved_jobs'][-5:]:  # Show last 5 saved jobs
            st.write(f"• **{job['title']}** at **{job['company']}** - [View]({job['url']})")

if __name__ == "__main__":
    main()
