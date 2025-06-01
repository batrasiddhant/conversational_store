import { type NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const { query, filters } = await request.json();

    if (!query) {
      return NextResponse.json({ error: "Query is required" }, { status: 400 });
    }

    // Call your FastAPI backend
    const response = await fetch("http://127.0.0.1:8000/api/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, filters }),
    });

    if (!response.ok) {
      // Forward error from backend
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.error || "FastAPI backend error" },
        { status: response.status }
      );
    }

    // Get JSON data from FastAPI
    const data = await response.json();

    // Return the FastAPI response directly
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error calling FastAPI backend:", error);
    return NextResponse.json(
      { error: "Failed to process search request" },
      { status: 500 }
    );
  }
}




// import { type NextRequest, NextResponse } from "next/server"

// // This would be replaced with actual search logic using the RAG pipeline
// export async function POST(request: NextRequest) {
//   try {
//     const { query, filters } = await request.json()

//     if (!query) {
//       return NextResponse.json({ error: "Query is required" }, { status: 400 })
//     }

//     // Simulate processing time
//     await new Promise((resolve) => setTimeout(resolve, 1500))

//     // Mock search results
//     const categories = ["Cleansers", "Moisturizers", "Serums", "Masks", "Toners"]
//     let category = ""

//     if (query.toLowerCase().includes("serum")) {
//       category = "Serums"
//     } else if (query.toLowerCase().includes("moisturizer")) {
//       category = "Moisturizers"
//     } else if (query.toLowerCase().includes("cleanser")) {
//       category = "Cleansers"
//     } else if (query.toLowerCase().includes("mask")) {
//       category = "Masks"
//     } else if (query.toLowerCase().includes("toner")) {
//       category = "Toners"
//     } else {
//       category = categories[Math.floor(Math.random() * categories.length)]
//     }

//     const mockResults = Array(6)
//       .fill(null)
//       .map((_, i) => ({
//         id: i + 1,
//         name: `${category.slice(0, -1)} ${i + 1}`,
//         category,
//         price: Math.floor(Math.random() * 30) + 20,
//         margin: Math.floor(Math.random() * 40) + 30,
//         image: `/placeholder.svg?height=200&width=200`,
//         description: `A premium ${category.toLowerCase().slice(0, -1)} for all skin types.`,
//         justification: `This ${category.toLowerCase().slice(0, -1)} is perfect for your needs based on your query.`,
//       }))

//     // Sort by margin (highest first)
//     const sortedResults = mockResults.sort((a, b) => b.margin - a.margin)

//     return NextResponse.json({
//       results: sortedResults,
//       summary: `Here are some ${category.toLowerCase()} that match your needs.`,
//     })
//   } catch (error) {
//     console.error("Error processing search request:", error)
//     return NextResponse.json({ error: "Failed to process search request" }, { status: 500 })
//   }
// }
