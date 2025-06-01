import { type NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const { query } = await request.json();

    if (!query) {
      return NextResponse.json({ error: "Query is required" }, { status: 400 });
    }

    // Call your FastAPI backend
    const response = await fetch("http://127.0.0.1:8000/api/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.error || "FastAPI backend error" },
        { status: response.status }
      );
    }

    const data = await response.json();

    // Return FastAPI's response
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error calling FastAPI backend:", error);
    return NextResponse.json(
      { error: "Failed to process chat request" },
      { status: 500 }
    );
  }
}




// import { type NextRequest, NextResponse } from "next/server"

// // This would be replaced with actual LLM integration
// async function mockLLMResponse(query: string) {
//   // Simulate processing time
//   await new Promise((resolve) => setTimeout(resolve, 1000))

//   if (query.toLowerCase().includes("serum") || query.toLowerCase().includes("moisturizer")) {
//     return {
//       response: `Based on your interest in ${query.toLowerCase().includes("serum") ? "serums" : "moisturizers"}, would you prefer products for sensitive, oily, or dry skin?`,
//       needsFollowUp: true,
//       products: [],
//     }
//   } else if (query.toLowerCase().includes("sensitive") || query.toLowerCase().includes("gentle")) {
//     return {
//       response:
//         "I've found some gentle products for sensitive skin. Would you prefer a cleanser, moisturizer, or serum?",
//       needsFollowUp: true,
//       products: [],
//     }
//   } else if (query.toLowerCase().includes("is") || query.toLowerCase().includes("how")) {
//     return {
//       response:
//         "Based on our product information and customer reviews, our Vitamin C serum is suitable for most skin types, including sensitive skin when used as directed. Many customers with sensitive skin have reported positive results without irritation.",
//       needsFollowUp: false,
//       products: [],
//     }
//   } else {
//     return {
//       response:
//         "I'd like to help you find the right skincare products. Could you tell me your skin type (dry, oily, combination, sensitive) and what specific concerns you're looking to address?",
//       needsFollowUp: true,
//       products: [],
//     }
//   }
// }

// export async function POST(request: NextRequest) {
//   try {
//     const { query } = await request.json()

//     if (!query) {
//       return NextResponse.json({ error: "Query is required" }, { status: 400 })
//     }

//     const result = await mockLLMResponse(query)

//     return NextResponse.json(result)
//   } catch (error) {
//     console.error("Error processing chat request:", error)
//     return NextResponse.json({ error: "Failed to process request" }, { status: 500 })
//   }
// }
