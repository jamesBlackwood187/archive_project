#r @"libs/MongoDB.Bson.dll"
#r @"libs/MongoDB.Driver.dll"
#r @"libs/MongoDB.Driver.Core.dll"
#r @"libs/MongoDB.Driver.Legacy.dll"
#r @"libs/MongoDB.FSharp.dll"
open MongoDB.Bson
open MongoDB.Driver
open MongoDB.FSharp

let connectionString = "mongodb://localhost"
let client = new MongoClient(connectionString)

let db = client.GetDatabase("bitcoin")

let getAllTrades (database:IMongoDatabase) =  
  let collection = database.GetCollection<BsonDocument> "trades"

  collection.FindAll()
  |> Seq.iter (printfn "%A")

let trades = getAllTrades db