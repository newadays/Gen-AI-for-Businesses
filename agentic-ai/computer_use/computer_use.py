#!/usr/bin/env python
# coding: utf-8

import os
import requests
import json
from dotenv import load_dotenv
import base64
from IPython.display import Markdown
from playwright.async_api import async_playwright, Page
import asyncio
import json
import re

load_dotenv()

# Define the base URL
BASE_URL = "https://api.llama.com/v1"

imagePath = "sample_screenshot.png"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# --- 2Captcha Solver (if API key is available) ---
solver = None
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
TWOCAPTCHA_API_KEY = os.getenv("TWOCAPTCHA_API_KEY")

if TWOCAPTCHA_API_KEY:
    try:
        from twocaptcha import TwoCaptcha
        solver = TwoCaptcha(TWOCAPTCHA_API_KEY)
        print("2Captcha solver initialized.")
    except ImportError:
        print("WARNING: 'twocaptcha' library not found. Please install it for reCAPTCHA handling.")
        solver = None
    except Exception as e:
        print(f"Error initializing 2Captcha solver: {e}")
        solver = None

def chat_completion(messages, model="Llama-4-Scout-17B-16E-Instruct-FP8", max_tokens=256):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLAMA_API_KEY}",
    }

    payload = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    return response.json()

def parse_accessibility_tree(node, indent=0):
    """
    Recursively parses the accessibility tree and prints a readable structure.
    """
    res = ""

    def _parse_node(node, indent, res):
        if not node or "role" not in node:
            return res

        indented_space = " " * indent

        if "value" in node:
            res = (
                res
                + f"{indented_space}Role: {node['role']} - Name: {node.get('name', 'No name')} - Value: {node['value']}\n"
            )
        else:
            res = (
                res
                + f"{indented_space}Role: {node['role']} - Name: {node.get('name', 'No name')}\n"
            )

        if "children" in node:
            for child in node["children"]:
                res = _parse_node(child, indent + 2, res)

        return res

    return _parse_node(node, indent, res)

# Planning and execution prompts (keeping them the same as in your original code)
planning_prompt = """
Given a user request, define a very simple plan of subtasks (actions) to achieve the desired outcome and execute them iteratively using Playwright.

1. Understand the Task:
   - Interpret the user's request and identify the core goal.
   - Break down the task into a few smaller, actionable subtasks to achieve the goal effectively.

2. Planning Actions:
   - Translate the user's request into a high-level plan of actions.
   - Example actions include:
     - Searching for specific information.
     - Navigating to specified URLs.
     - Interacting with website elements (clicking, filling).
     - Extracting or validating data.

Input:
- User Request (Task)

Output from the Agent:
- Step-by-Step Action Plan:: Return only an ordered list of actions. Only return the list, no other text.

**Example User Requests and Agent Behavior:**

1. **Input:** "Search for a product on Amazon."
   - **Output:**
     1. Navigate to Amazon's homepage.
     2. Enter the product name in the search bar and perform the search.
     3. Extract and display the top results, including the product title, price, and ratings.

2. **Input:** "Find the cheapest flight to Tokyo."
   - **Output:**
     1. Visit a flight aggregator website (e.g. google flights).
     2. Enter the departure city.
     3. Enter the destination city
     4. Enter the departure date
     5. Enter the return date
     6. Click the 'Done' button to confirm departure and return dates.
     6. Click the 'Search' button to find available flights.
     7. Extract and compare the flight options, highlighting the cheapest option.

3. **Input:** "Buy tickets for the next Warriors game."
   - **Output:**
     1. Navigate to a ticket-selling platform (e.g., Ticketmaster).
     2. Fill the search bar with the team name.
     2. Search for upcoming team games.
     3. Select the next available game and purchase tickets for the specified quantity.
"""

execution_prompt = """
You will be given a task, a website's page accessibility tree, and the page screenshot as context. The screenshot is where you are now, use it to understand the accessibility tree. Based on that information, you need to decide the next step action. ONLY RETURN THE NEXT STEP ACTION IN A SINGLE JSON.

When selecting elements, use elements from the accessibility tree.

Reflect on what you are seeing in the accessibility tree and the screenshot and decide the next step action, elaborate on it in reasoning, and choose the next appropriate action.

Selectors must follow the format:
- For a button with a specific name: "button=ButtonName"
- For a placeholder (e.g., input field): "placeholder=PlaceholderText"
- For text: "text=VisibleText"

Make sure to analyze the accessibility tree and the screenshot to understand the current state, if something is not clear, you can use the previous actions to understand the current state. Explain why you are in the current state in current_state.

You will be given a task and you MUST return the next step action in JSON format:
{
    "current_state": "Where are you now? Analyze the accessibility tree and the screenshot to understand the current state.",
    "reasoning": "What is the next step to accomplish the task?",
    "action": "navigation" or "click" or "fill" or "solve_recaptcha" or "finished",
    "url": "https://www.example.com", // Only for navigation actions
    "selector": "button=Click me", // For click or fill actions, derived from the accessibility tree. For solve_recaptcha, this will be "recaptcha"
    "value": "Input text", // Only for fill actions
    "sitekey": "YOUR_RECAPTCHA_SITEKEY", // Only for solve_recaptcha actions, required by 2Captcha
    "page_url": "https://www.example.com/captcha", // Only for solve_recaptcha actions, required by 2Captcha
}

### Guidelines:
1. Use **"navigation"** for navigating to a new website through a URL.
2. Use **"click"** for interacting with clickable elements. Examples:
   - Buttons: "button=Click me"
   - Text: "text=VisibleText"
   - Placeholders: "placeholder=Search..."
   - Link: "link=BUY NOW"
3. Use **"fill"** for inputting text into editable fields. Examples:
   - Placeholder: "placeholder=Search..."
   - Textbox: "textbox=Flight destination output"
   - Input: "input=Search..."
4. Use **"solve_recaptcha"** when you detect a reCAPTCHA challenge. YOU MUST DETECT THE RECAPTCHA FIRST. IF A RECAPTCHA IS PRESENT, THIS IS YOUR IMMEDIATE NEXT ACTION.
   - You MUST provide the `sitekey` and `page_url`. The `selector` should be "recaptcha".
   - **How to find sitekey:** Look for `data-sitekey="..."` attribute, typically within a `div` element with `class="g-recaptcha"` or an `iframe` with `title="reCAPTCHA"`. You might need to inspect the HTML (`await page.content()`) if it's not directly visible in the accessibility tree name.
   - Example `solve_recaptcha` action:
     ```json
     {
         "current_state": "Encountered a reCAPTCHA challenge.",
         "reasoning": "A reCAPTCHA challenge is blocking progress. I must solve it to proceed. I will extract the sitekey from the 'g-recaptcha' element and use the current page URL.",
         "action": "solve_recaptcha",
         "selector": "recaptcha",
         "sitekey": "6Le-wvkSAAAAAPBMRTg0Wt_EQMoJgFeyG28PSUZM", 
         "page_url": "https://www.example.com/login" 
     }
5. Use **"finished"** when the task is done. For example:
   - If a task is successfully completed.
   - If navigation confirms you are on the correct page.

### Accessibility Tree Examples:
You will be given an accessibility tree to interact with the webpage. It consists of a nested node structure that represents elements on the page. For example:

Role: generic - Name: 
   Role: text - Name: Phoenix
   Role: button - Name: 
   Role: listitem - Name: 
   Role: textbox - Name: Where from?
Role: button - Name: Swap where to and where from
Role: generic - Name: 
   Role: textbox - Name: Where to?
Role: textbox - Name: Return
Role: button - Name: 
Role: button - Name: 
Role: textbox - Name: Departure
Role: button - Name: 
Role: button - Name: Done
Role: button - Name: Search

This section indicates that there is a textbox with a name "Where to?" filled with Phoenix. There is also a button with the name "Swap where to and where from". Another textbox with the name "where to?" not filled with any text. There are also textboxes with the names "Departure", "Return", which are not filled with any dates, and a buttons named "Done" and "Search".

Retry actions at most 2 times before trying a different action.

### Examples:
1. To click on a button labeled "Search":
   {
       "current_state": "On the homepage of a search engine.",
       "reasoning": "The accessibility tree shows a button named 'Search'. Clicking it is the appropriate next step to proceed with the task.",
       "action": "click",
       "selector": "button=Search"
   }

2. To fill a search bar with the text "AI tools":
   {
       "current_state": "On the search page with a focused search bar.",
       "reasoning": "The accessibility tree shows an input field with placeholder 'Search...'. Entering the query 'AI tools' fulfills the next step of the task.",
       "action": "fill",
       "selector": "placeholder=Search...",
       "value": "AI tools"
   }

3. To navigate to a specific URL:
   {
       "current_state": "Starting from a blank page.",
       "reasoning": "The task requires visiting a specific website to gather relevant information. Navigating to the URL is the first step.",
       "action": "navigation",
       "url": "https://example.com"
   }

4. To finish the task:
   {
       "current_state": "Completed the search and extracted the necessary data.",
       "reasoning": "The task goal has been achieved, and no further actions are required.",
       "action": "finished"
   }
"""

# --- Enhanced reCAPTCHA Detection and Solving Functions ---

async def detect_recaptcha_type_and_sitekey(page: Page):
    """
    Enhanced reCAPTCHA detection that properly identifies v2 vs v3 and extracts sitekey
    """
    try:
        html_content = await page.content()
        
        # Check for v2 indicators first (more explicit)
        v2_indicators = [
            'class="g-recaptcha"',
            'div[data-sitekey]',
            'iframe[src*="recaptcha/api2"]',
            'grecaptcha.render'
        ]
        
        # Check for v3 indicators
        v3_indicators = [
            'grecaptcha.execute',
            'grecaptcha.ready',
            'data-action=',
            'recaptcha/api.js'
        ]
        
        is_v2 = any(indicator in html_content for indicator in v2_indicators)
        is_v3 = any(indicator in html_content for indicator in v3_indicators)
        
        # Extract sitekey using comprehensive patterns
        sitekey_patterns = [
            r'data-sitekey\s*=\s*["\']([^"\']+)["\']',  # data-sitekey="value"
            r'sitekey\s*:\s*["\']([^"\']+)["\']',        # sitekey: 'value'
            r'grecaptcha\.render\([^,]*,\s*\{\s*["\']?sitekey["\']?\s*:\s*["\']([^"\']+)["\']',  # grecaptcha.render
            r'grecaptcha\.execute\s*\(\s*["\']([^"\']+)["\']',  # grecaptcha.execute('sitekey'
            r'["\']sitekey["\']\s*:\s*["\']([^"\']+)["\']',     # "sitekey": "value"
        ]
        
        sitekey = None
        for pattern in sitekey_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                sitekey = match.group(1)
                print(f"Found sitekey using pattern: {pattern[:30]}...")
                break
        
        # Try to get sitekey from DOM elements
        if not sitekey:
            try:
                # Check for data-sitekey attribute in DOM
                sitekey_element = await page.query_selector('[data-sitekey]')
                if sitekey_element:
                    sitekey = await sitekey_element.get_attribute('data-sitekey')
                    print(f"Found sitekey from DOM element: {sitekey}")
            except Exception as e:
                print(f"Error getting sitekey from DOM: {e}")
        
        # Determine version priority
        if is_v2 and not is_v3:
            version = "v2"
        elif is_v3 and not is_v2:
            version = "v3"
        elif is_v2 and is_v3:
            # Both present, check for explicit v2 elements
            v2_elements = await page.query_selector_all('.g-recaptcha, iframe[src*="api2"]')
            version = "v2" if v2_elements else "v3"
        else:
            # Default to v3 if indicators are unclear
            version = "v3"
        
        print(f"Detected reCAPTCHA version: {version}, sitekey: {sitekey}")
        return version, sitekey
        
    except Exception as e:
        print(f"Error detecting reCAPTCHA: {e}")
        return None, None

async def solve_recaptcha_v2(page: Page, sitekey: str, solver):
    """
    Improved reCAPTCHA v2 solving with better error handling
    """
    try:
        print("Solving reCAPTCHA v2...")
        current_url = page.url
        
        if not sitekey:
            print("Error: No sitekey provided for v2 solving")
            return False
        
        print(f"Submitting v2 reCAPTCHA to 2Captcha with sitekey: {sitekey}")
        
        # Submit to 2Captcha with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(solver.recaptcha, sitekey=sitekey, url=current_url, version="v2"),
            timeout=120  # Increased timeout for v2
        )
        
        if result and "code" in result:
            captcha_response = result["code"]
            print(f"Received v2 solution: {captcha_response[:20]}...")
            
            # Inject solution into page
            await page.evaluate(f"""
                // Set the response in the standard textarea
                const responseTextarea = document.getElementById('g-recaptcha-response');
                if (responseTextarea) {{
                    responseTextarea.value = '{captcha_response}';
                    responseTextarea.style.display = 'block';
                }}
                
                // Also set in any hidden inputs
                const hiddenInputs = document.querySelectorAll('input[name="g-recaptcha-response"]');
                hiddenInputs.forEach(input => input.value = '{captcha_response}');
                
                // Override grecaptcha methods if available
                if (typeof grecaptcha !== 'undefined') {{
                    if (grecaptcha.getResponse) {{
                        const originalGetResponse = grecaptcha.getResponse;
                        grecaptcha.getResponse = function(widgetId) {{
                            return '{captcha_response}';
                        }};
                    }}
                }}
                
                // Trigger any callback events
                const recaptchaDiv = document.querySelector('.g-recaptcha');
                if (recaptchaDiv) {{
                    const event = new Event('change', {{ bubbles: true }});
                    recaptchaDiv.dispatchEvent(event);
                }}
            """)
            
            print("reCAPTCHA v2 solution injected successfully")
            await asyncio.sleep(2)
            return True
        else:
            print("No valid response received from 2Captcha for v2")
            return False
            
    except asyncio.TimeoutError:
        print("Timeout while solving reCAPTCHA v2")
        return False
    except Exception as e:
        print(f"Error solving reCAPTCHA v2: {e}")
        return False

async def solve_recaptcha_v3(page: Page, sitekey: str, solver):
    """
    Improved reCAPTCHA v3 solving with better action detection
    """
    try:
        print("Solving reCAPTCHA v3...")
        current_url = page.url
        
        if not sitekey:
            print("Error: No sitekey provided for v3 solving")
            return False
        
        # Extract action from page content
        html_content = await page.content()
        action_patterns = [
            r'data-action\s*=\s*["\']([^"\']+)["\']',
            r'grecaptcha\.execute\([^,]+,\s*\{\s*["\']?action["\']?\s*:\s*["\']([^"\']+)["\']',
            r'["\']action["\']\s*:\s*["\']([^"\']+)["\']'
        ]
        
        action = "verify"  # Default action
        for pattern in action_patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                action = match.group(1)
                print(f"Found action: {action}")
                break
        
        print(f"Submitting v3 reCAPTCHA to 2Captcha with sitekey: {sitekey}, action: {action}")
        
        # Submit to 2Captcha
        result = await asyncio.wait_for(
            asyncio.to_thread(solver.recaptcha, sitekey=sitekey, url=current_url, version="v3", action=action),
            timeout=120
        )
        
        if result and "code" in result:
            captcha_response = result["code"]
            print(f"Received v3 solution: {captcha_response[:20]}...")
            
            # Inject solution into page
            await page.evaluate(f"""
                // Override grecaptcha.execute to return our token
                if (typeof grecaptcha !== 'undefined') {{
                    grecaptcha.ready(function() {{
                        const originalExecute = grecaptcha.execute;
                        grecaptcha.execute = function(siteKey, options) {{
                            console.log('Intercepted grecaptcha.execute call');
                            if (siteKey === '{sitekey}') {{
                                return Promise.resolve('{captcha_response}');
                            }}
                            return originalExecute.call(this, siteKey, options);
                        }};
                    }});
                }}
                
                // Set in any token input fields
                const tokenInputs = document.querySelectorAll(
                    'input[name="g-recaptcha-response"], input[name*="recaptcha"], input[name*="captcha"]'
                );
                tokenInputs.forEach(input => {{
                    input.value = '{captcha_response}';
                    console.log('Set token in input:', input.name);
                }});
                
                // Set in any hidden divs or spans that might hold the token
                const tokenContainers = document.querySelectorAll('[data-recaptcha-token], [id*="recaptcha-token"]');
                tokenContainers.forEach(container => {{
                    container.setAttribute('data-recaptcha-token', '{captcha_response}');
                    container.textContent = '{captcha_response}';
                }});
                
                // Store token globally for form submission
                window.recaptchaToken = '{captcha_response}';
            """)
            
            print("reCAPTCHA v3 solution injected successfully")
            await asyncio.sleep(2)
            return True
        else:
            print("No valid response received from 2Captcha for v3")
            return False
            
    except asyncio.TimeoutError:
        print("Timeout while solving reCAPTCHA v3")
        return False
    except Exception as e:
        print(f"Error solving reCAPTCHA v3: {e}")
        return False

async def handle_recaptcha(page: Page, current_url: str, recaptcha_sitekey: str = None):
    """
    Main reCAPTCHA handler with improved detection and solving
    """
    if not solver:
        print("Error: 2Captcha solver not initialized")
        return False
    
    # Auto-detect if no sitekey provided
    if not recaptcha_sitekey:
        print("No sitekey provided, attempting auto-detection...")
        version, detected_sitekey = await detect_recaptcha_type_and_sitekey(page)
        if not detected_sitekey:
            print("Could not detect reCAPTCHA sitekey automatically")
            return False
        recaptcha_sitekey = detected_sitekey
    else:
        # Detect version even if sitekey is provided
        version, _ = await detect_recaptcha_type_and_sitekey(page)
    
    print(f"Attempting to solve reCAPTCHA {version} with sitekey: {recaptcha_sitekey}")
    
    if version == "v2":
        return await solve_recaptcha_v2(page, recaptcha_sitekey, solver)
    else:  # Default to v3
        return await solve_recaptcha_v3(page, recaptcha_sitekey, solver)

# Define your task here
task = "Find the cheapest round trip flight from New York to Lagos"
# task = "Find a birthday present for my daughter of age 12"

async def run_browser():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False, channel="chrome")
        page = await browser.new_page()
        
        # Generate plan
        print("Generating plan...")
        planning_response = chat_completion(
            messages=[
                {"role": "system", "content": planning_prompt},
                {"role": "user", "content": task},
            ],
        )
        
        plan = planning_response["completion_message"]["content"]["text"]
        print(plan)
        steps = [line.strip()[3:] for line in plan.strip().split("\n")]
        
        await asyncio.sleep(1)
        await page.goto("https://google.com/")
        previous_actions = []
        
        try:
            while True:
                # Get context from page
                accessibility_tree = await page.accessibility.snapshot()
                accessibility_tree = parse_accessibility_tree(accessibility_tree)
                await page.screenshot(path="screenshot.png")
                base64_image = encode_image("screenshot.png")
                
                # Get LLM response
                response = chat_completion(
                    messages=[
                        {"role": "system", "content": execution_prompt},
                        {
                            "role": "system",
                            "content": f"Plan to execute: {steps}\n\nAccessibility Tree: {accessibility_tree}\n\nPrevious actions: {previous_actions}",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"What should be the next action to accomplish the task: {task} based on the current state? Remember to review the plan and select the next action based on the current state. Provide the next action in JSON format strictly as specified above.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        },
                    ],
                )
                
                res = response["completion_message"]["content"]["text"]
                res = res.rstrip(",")
                res = "".join(c for c in res if ord(c) >= 32 or ord(c) == 10 or ord(c) == 13)
                print("Agent response:", res)
                
                try:
                    match = re.search(r"\{.*\}", res, re.DOTALL)
                    if match:
                        output = json.loads(match.group(0))
                except Exception as e:
                    print("Error parsing JSON:", e)
                    continue
                
                # Detect reCAPTCHA in current state description
                if "reCAPTCHA" in output.get("current_state", "") or "captcha" in output.get("current_state", "").lower():
                    output["action"] = "solve_recaptcha"
                
                # Execute actions
                if output["action"] == "navigation":
                    try:
                        await page.goto(output["url"])
                        previous_actions.append(f"navigated to {output['url']}, SUCCESS")
                    except Exception as e:
                        previous_actions.append(f"Error navigating to {output['url']}: {e}")
                
                elif output["action"] == "click":
                    try:
                        selector_parts = output["selector"].split("=", 1)
                        if len(selector_parts) == 2:
                            selector_type, selector_name = selector_parts
                            await page.get_by_role(selector_type, name=selector_name).first.click()
                            previous_actions.append(f"clicked {output['selector']}, SUCCESS")
                        else:
                            previous_actions.append(f"Invalid selector format: {output['selector']}")
                    except Exception as e:
                        previous_actions.append(f"Error clicking on {output['selector']}: {e}")
                
                elif output["action"] == "fill":
                    try:
                        selector_parts = output["selector"].split("=", 1)
                        if len(selector_parts) == 2:
                            selector_type, selector_name = selector_parts
                            await page.get_by_role(selector_type, name=selector_name).fill(output["value"])
                            await asyncio.sleep(1)
                            await page.keyboard.press("Enter")
                            previous_actions.append(f"filled {output['selector']} with {output['value']}, SUCCESS")
                        else:
                            previous_actions.append(f"Invalid selector format: {output['selector']}")
                    except Exception as e:
                        previous_actions.append(f"Error filling {output['selector']} with {output['value']}: {e}")
                
                elif output["action"] == "solve_recaptcha":
                    current_url = page.url
                    recaptcha_sitekey = output.get("sitekey")
                    
                    print(f"Attempting to solve reCAPTCHA...")
                    success = await handle_recaptcha(page, current_url, recaptcha_sitekey)
                    
                    if success:
                        previous_actions.append("reCAPTCHA solved successfully")
                    else:
                        previous_actions.append("Failed to solve reCAPTCHA")
                
                elif output["action"] == "finished":
                    print("Task completed!")
                    if "summary" in output:
                        print(output["summary"])
                    break
                
                await asyncio.sleep(1)
                
                # Wait for user input
                user_input = input("Press 'q' to quit or Enter to continue: ")
                if user_input.lower() == "q":
                    break
        
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            await browser.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_browser())