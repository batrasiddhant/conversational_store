import { type NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");

    const fastapiUrl = new URL("http://ec2-13-235-74-126.ap-south-1.compute.amazonaws.com:8142/api/products");

    if (category) {
      fastapiUrl.searchParams.set("category", category);
    }

    const response = await fetch(fastapiUrl.toString());

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.error || "FastAPI error" },
        { status: response.status }
      );
    }

    const products = await response.json();
    return NextResponse.json(products);
  } catch (error) {
    console.error("Error fetching products:", error);
    return NextResponse.json({ error: "Failed to fetch products" }, { status: 500 });
  }
}




// import { type NextRequest, NextResponse } from "next/server"

// // This would be replaced with actual product data from the Excel file
// const mockProducts = Array(30)
//   .fill(null)
//   .map((_, i) => {
//     const categories = ["Cleansers", "Moisturizers", "Serums", "Masks", "Toners"]
//     const randomCategory = categories[Math.floor(Math.random() * categories.length)]

//     return {
//       id: i + 1,
//       name: `${randomCategory.slice(0, -1)} ${i + 1}`,
//       category: randomCategory,
//       price: Math.floor(Math.random() * 30) + 10,
//       margin: Math.floor(Math.random() * 40) + 30,
//       image: `/placeholder.svg?height=200&width=200`,
//       description: `A premium skincare product for all skin types.`,
//     }
//   })

// export async function GET(request: NextRequest) {
//   try {
//     const { searchParams } = new URL(request.url)
//     const category = searchParams.get("category")

//     let filteredProducts = mockProducts

//     if (category) {
//       filteredProducts = mockProducts.filter((p) => p.category === category)
//     }

//     return NextResponse.json(filteredProducts)
//   } catch (error) {
//     console.error("Error fetching products:", error)
//     return NextResponse.json({ error: "Failed to fetch products" }, { status: 500 })
//   }
// }
