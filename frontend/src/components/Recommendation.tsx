interface RecommendationProps {
  recommendation: string;
}

const Recommendation = ({ recommendation }: RecommendationProps) => {
  const color = recommendation === 'buy' ? 'text-green-600' : recommendation === 'sell' ? 'text-red-600' : 'text-yellow-600';
  return (
    <div className="mb-4">
      <h2 className="text-xl font-bold mb-2">Recommendation</h2>
      <p className={`text-2xl font-bold ${color}`}>{recommendation.toUpperCase()}</p>
    </div>
  );
};

export default Recommendation;
