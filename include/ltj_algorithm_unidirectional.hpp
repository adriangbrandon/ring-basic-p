/*
 * ltj_algorithm.hpp
 * Copyright (C) 2020 Author removed for double-blind evaluation
 *
 *
 * This is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */



#ifndef RING_LTJ_ALGORITHM_UNIDIRECTIONAL_HPP
#define RING_LTJ_ALGORITHM_UNIDIRECTIONAL_HPP


#include <triple_pattern.hpp>
#include <uring.hpp>
#include <ltj_iterator_unidirectional.hpp>
#include <veo_simple.hpp>
#include <veo_adaptive.hpp>
#include <wt_intersection_iterator.hpp>
#include <results_collector.hpp>

namespace ring {

    template<class iterator_t = ltj_iterator_unidirectional<uring<>, uint8_t, uint64_t>,
             class veo_t = veo::veo_adaptive<iterator_t, util::trait_size> >
    class ltj_algorithm_unidirectional {

    public:
        typedef uint64_t value_type;
        typedef uint64_t size_type;
        typedef iterator_t ltj_iter_type;
        typedef typename ltj_iter_type::var_type var_type;
        typedef typename ltj_iter_type::ring_type ring_type;
        typedef typename ltj_iter_type::value_type const_type;
        typedef veo_t veo_type;
        typedef std::unordered_map<var_type, std::vector<ltj_iter_type*>> var_to_iterators_type;
        typedef std::vector<std::pair<var_type, value_type>> tuple_type;
        typedef std::chrono::high_resolution_clock::time_point time_point_type;
        typedef typename ring_type::bwt_type::wm_type wm_type;
        typedef ::util::results_collector<tuple_type> results_type;

    private:
        const std::vector<triple_pattern>* m_ptr_triple_patterns;
        veo_type m_veo;
        ring_type* m_ptr_ring;
        std::vector<ltj_iter_type> m_iterators;
        var_to_iterators_type m_var_to_iterators;
        bool m_is_empty = false;


        void copy(const ltj_algorithm_unidirectional &o) {
            m_ptr_triple_patterns = o.m_ptr_triple_patterns;
            m_veo = o.m_veo;
            m_ptr_ring = o.m_ptr_ring;
            m_iterators = o.m_iterators;
            m_var_to_iterators = o.m_var_to_iterators;
            m_is_empty = o.m_is_empty;
        }


        inline void add_var_to_iterator(const var_type var, ltj_iter_type* ptr_iterator){
            auto it =  m_var_to_iterators.find(var);
            if(it != m_var_to_iterators.end()){
                it->second.push_back(ptr_iterator);
            }else{
                std::vector<ltj_iter_type*> vec = {ptr_iterator};
                m_var_to_iterators.insert({var, vec});
            }
        }

    public:


        ltj_algorithm_unidirectional() = default;

        ltj_algorithm_unidirectional(const std::vector<triple_pattern>* triple_patterns, ring_type* ring){

            m_ptr_triple_patterns = triple_patterns;
            m_ptr_ring = ring;

            size_type i = 0;
            m_iterators.reserve(m_ptr_triple_patterns->size());
            for(const auto& triple : *m_ptr_triple_patterns){
                //Bulding iterators
                m_iterators.emplace_back(ltj_iter_type(&triple, m_ptr_ring));
                if(m_iterators[i].is_empty()){
                    m_is_empty = true;
                    return;
                }

                //For each variable we add the pointers to its iterators
                if(triple.o_is_variable()){
                    add_var_to_iterator(triple.term_o.value, &(m_iterators[i]));
                }
                if(triple.p_is_variable()){
                    add_var_to_iterator(triple.term_p.value, &(m_iterators[i]));
                }
                if(triple.s_is_variable()){
                    add_var_to_iterator(triple.term_s.value, &(m_iterators[i]));
                }
                ++i;
            }

            m_veo = veo_type(m_ptr_triple_patterns, &m_iterators, &m_var_to_iterators, m_ptr_ring);

        }

        //! Copy constructor
        ltj_algorithm_unidirectional(const ltj_algorithm_unidirectional &o) {
            copy(o);
        }

        //! Move constructor
        ltj_algorithm_unidirectional(ltj_algorithm_unidirectional &&o) {
            *this = std::move(o);
        }

        //! Copy Operator=
        ltj_algorithm_unidirectional &operator=(const ltj_algorithm_unidirectional &o) {
            if (this != &o) {
                copy(o);
            }
            return *this;
        }

        //! Move Operator=
        ltj_algorithm_unidirectional &operator=(ltj_algorithm_unidirectional &&o) {
            if (this != &o) {
                m_ptr_triple_patterns = std::move(o.m_ptr_triple_patterns);
                m_veo = std::move(o.m_veo);
                m_ptr_ring = std::move(o.m_ptr_ring);
                m_iterators = std::move(o.m_iterators);
                m_var_to_iterators = std::move(o.m_var_to_iterators);
                m_is_empty = o.m_is_empty;
            }
            return *this;
        }

        void swap(ltj_algorithm_unidirectional &o) {
            std::swap(m_ptr_triple_patterns, o.m_ptr_triple_patterns);
            std::swap(m_veo, o.m_veo);
            std::swap(m_ptr_ring, o.m_ptr_ring);
            std::swap(m_iterators, o.m_iterators);
            std::swap(m_var_to_iterators, o.m_var_to_iterators);
            std::swap(m_is_empty, o.m_is_empty);
        }


        /**
        *
        * @param res               Results
        * @param limit_results     Limit of results
        * @param timeout_seconds   Timeout in seconds
        */
        void join(results_type &res,
                  const size_type limit_results = 0, const size_type timeout_seconds = 0){
            if(m_is_empty) return;
            time_point_type start = std::chrono::high_resolution_clock::now();
            tuple_type t(m_veo.size());
            search(0, t, res, start, limit_results, timeout_seconds);
        };


        /**
         *
         * @param j                 Index of the variable
         * @param tuple             Tuple of the current search
         * @param res               Results
         * @param start             Initial time to check timeout
         * @param limit_results     Limit of results
         * @param timeout_seconds   Timeout in seconds
         */
        bool search(const size_type j, tuple_type &tuple, results_type &res,
                    const time_point_type start,
                    const size_type limit_results = 0, const size_type timeout_seconds = 0){

            //(Optional) Check timeout
            if(timeout_seconds > 0){
                time_point_type stop = std::chrono::high_resolution_clock::now();
                auto sec = std::chrono::duration_cast<std::chrono::seconds>(stop-start).count();
                if(sec > timeout_seconds) return false;
            }

            //(Optional) Check limit
            if(limit_results > 0 && res.size() == limit_results) return false;

            if(j == m_veo.size()){
                //Report results
                res.add(tuple);
            }else{
                var_type x_j = m_veo.next();
                std::vector<ltj_iter_type*>& itrs = m_var_to_iterators[x_j];
                bool ok;
                if(itrs.size() == 1 && itrs[0]->in_last_level()) {//Lonely variables
                    value_type c = itrs[0]->seek_last(x_j);
                    //auto results = itrs[0]->seek_all(x_j);
                    //std::cout << "Results: " << results.size() << std::endl;
                    //std::cout << "Seek (last level): (" << (uint64_t) x_j << ": size=" << results.size() << ")" <<std::endl;
                    while (c != 0) { //If empty c=0
                        //1. Adding result to tuple
                        tuple[j] = {x_j, c};
                        //2. Going down in the trie by setting x_j = c (\mu(t_i) in paper)
                        itrs[0]->down(x_j, c);
                        m_veo.down();
                        //2. Search with the next variable x_{j+1}
                        ok = search(j + 1, tuple, res, start, limit_results, timeout_seconds);
                        if(!ok) return false;
                        //4. Going up in the trie by removing x_j = c
                        itrs[0]->up(x_j);
                        m_veo.up();
                        c = itrs[0]->seek_last_next(x_j);
                    }
                }else {
                    wt_intersection_iterator<wm_type> wts_iterator(itrs, x_j);
                    value_type c = wts_iterator.next();
                    //value_type c = seek(x_j);
                    //std::cout << "Seek (init): (" << (uint64_t) x_j << ": " << c << ")" <<std::endl;
                    while (c != 0) { //If empty c=0
                        //1. Adding result to tuple
                        tuple[j] = {x_j, c};
                        //2. Going down in the tries by setting x_j = c (\mu(t_i) in paper)
                        for (ltj_iter_type* iter : itrs) {
                            iter->down(x_j, c);
                        }
                        m_veo.down();
                        //3. Search with the next variable x_{j+1}
                        ok = search(j + 1, tuple, res, start, limit_results, timeout_seconds);
                        if(!ok) return false;
                        //4. Going up in the tries by removing x_j = c
                        for (ltj_iter_type *iter : itrs) {
                            iter->up(x_j);
                        }
                        m_veo.up();
                        //5. Next constant for x_j
                        c = wts_iterator.next();
                        //c = seek(x_j, c + 1);
                        //std::cout << "Seek (bucle): (" << (uint64_t) x_j << ": " << c << ")" <<std::endl;
                    }
                }
                m_veo.done();
            }
            return true;
        };

        /**
         *
         * @param x_j   Variable
         * @param c     Constant. If it is unknown the value is -1
         * @return      The next constant that matches the intersection between the triples of x_j.
         *              If the intersection is empty, it returns 0.
         */

        value_type seek(const var_type x_j, value_type c=-1){
            std::vector<ltj_iter_type*>& itrs = m_var_to_iterators[x_j];
            value_type c_i, c_prev = 0, i = 0, n_ok = 0;
            while (true){
                //Compute leap for each triple that contains x_j
                if(c == -1){
                    c_i = itrs[i]->leap(x_j);
                }else{
                    c_i = itrs[i]->leap(x_j, c);
                }
                if(c_i == 0) return 0; //Empty intersection
                n_ok = (c_i == c_prev) ? n_ok + 1 : 1;
                if(n_ok == itrs.size()) return c_i;
                c = c_prev = c_i;
                i = (i+1 == itrs.size()) ? 0 : i+1;
            }
        }


        void print_veo(std::unordered_map<uint8_t, std::string> &ht){
            std::cout << "veo: " << std::endl;
            for(uint64_t j = 0; j < m_veo.size(); ++j){
                std::cout << "?" << ht[m_veo.next()] << " ";
            }
            std::cout << std::endl;
        }

        void print_query(std::unordered_map<uint8_t, std::string> &ht){
            std::cout << "Query: " << std::endl;
            for(size_type i = 0; i <  m_ptr_triple_patterns->size(); ++i){
                m_ptr_triple_patterns->at(i).print(ht);
                if(i < m_ptr_triple_patterns->size()-1){
                    std::cout << " . ";
                }
            }
            std::cout << std::endl;
        }

        void print_results(std::vector<tuple_type> &res, std::unordered_map<uint8_t, std::string> &ht){
            std::cout << "Results: " << std::endl;
            uint64_t i = 1;
            for(tuple_type &tuple :  res){
                std::cout << "[" << i << "]: ";
                for(std::pair<var_type, value_type> &pair : tuple){
                    std::cout << "?" << ht[pair.first] << "=" << pair.second << " ";
                }
                std::cout << std::endl;
                ++i;
            }
        }


    };
}

#endif //RING_LTJ_ALGORITHM_HPP